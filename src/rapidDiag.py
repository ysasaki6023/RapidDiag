import sys,os,h5py,glob,itertools,re
import numpy as np
import io

import keras
from keras.preprocessing import image as keras_image

from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras import Input
from keras import backend as K

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, mixture
import tensorflow as tf

from PIL import Image
import cv2

import tqdm

class rapidDiag(object):
    def __init__(self,platform="cui"):
        self.cmap = plt.get_cmap("tab10")
        self.comp = None
        self.image_size = (224,224,3)
        if platform=="cui":
            self.tqdm = tqdm.tqdm
        elif platform=="notebook":
            self.tqdm = tqdm.tqdm_notebook
        elif platform=="server":
            self.tqdm = lambda x:x

        return

    def set_model(self,model_type="mobilenet",model_layer="last"):
        if model_type=="vgg16":
            assert False, "vgg16 is not implemented yet"
            self.preprocess_input = keras.applications.vgg16.preprocess_input
            self.base_model = base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
            self.last_vis_layer = None 
        elif model_type=="mobilenet":
            self.preprocess_input = keras.applications.mobilenet.preprocess_input
            self.base_model = base_model = keras.applications.mobilenet.MobileNet(input_shape=self.image_size,weights='imagenet', include_top=False)
        else:
            assert False, "incorrect model_type"

        if model_layer=="last":
            h = base_model.output
        else:
            h = base_model.get_layer(model_layer).output

        self.last_vis_layer = last_vis_layer = h
        h = GlobalAveragePooling2D()(h)
        model = Model(inputs=base_model.input, outputs=h)
        self.output_dim = K.int_shape(h)[-1]
        self.model = model
        self.graph = tf.get_default_graph()

        # grad_cam

        self.v_input = v_input = Input(shape=(K.int_shape(model.output)[1],))
        #self.d0_input = d0_input = Input(shape=(K.int_shape(model.output)[1],))
        self.d1_input = d1_input = Input(shape=(K.int_shape(model.output)[1],))

        prob_d0 = K.dot(last_vis_layer-d1_input, K.transpose(-v_input))/K.sum(K.pow(-v_input,2))
        grads = K.gradients(prob_d0, last_vis_layer)[0]
        self.gradient_function = gradient_function = K.function([model.input,v_input,d1_input], [last_vis_layer,grads])

        return

    def grad_cam(self,img,v,d1):
        #img_orig = Image.open(img_path)
        img_orig = img
        img = img_orig.resize(self.image_size[:2],Image.BICUBIC)
        img = np.asarray(img).astype(np.float32)
        img.flags.writeable = True
        x_vec = np.array([img])
        x = self.preprocess_input(x_vec)
        with self.graph.as_default():
            output, grads_val = self.gradient_function([x,[v],[d1]])
        output,grads_val = output[0],grads_val[0]

        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        cam = cv2.resize(cam, img_orig.size, cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        
        return img_orig,cam

    def process_one(self,img,batch_size=64):
        """
        img: PIL Image形式の画像
        """
        # TODO: カラー以外の形式への対応
        img = img.resize(self.image_size[:2],Image.BICUBIC)
        img = np.asarray(img).astype(np.float32)
        img.flags.writeable = True
        x_vec = np.array([img])
        x = self.preprocess_input(x_vec)
        with self.graph.as_default():
            features = self.model.predict_on_batch(x)
        return features

    def process(self,batch_size=64,image_size=(224,224)):
        self.fpath_allcls = fpath_allcls = {}
        self.features_allcls = features_allcls = {}
        for cls in self.tqdm(self.target_files,desc="category"):
            #fpath_all = glob.glob(self.target_files[cls],recursive=True)
            fpath_all = glob.glob(self.target_files[cls],recursive=True)[:batch_size]
            fpath_allcls[cls] = fpath_all
            features_all = []
            for idx0 in self.tqdm(range(0,len(fpath_all),batch_size),desc=cls,leave=False):
                x_vec = [keras_image.img_to_array(keras_image.load_img(img_path, target_size=image_size, interpolation="bicubic")) for img_path in fpath_all[idx0:idx0+batch_size]]
                x_vec = np.array(x_vec)
                x = self.preprocess_input(x_vec)
                features = self.model.predict(x)
                for f in features:
                    features_all.append(f)
            features_all = np.array(features_all)
            features_allcls[cls] = features_all

        self.X = np.concatenate([features_allcls[c] for c in features_allcls],axis=0)
        self.cls_list = list(features_allcls.keys())
        self.Y = np.concatenate([np.array([self.cls_list.index(c)]*features_allcls[c].shape[0]) for c in features_allcls],axis=0)

        return

    def set_files(self):
        self.target_files = target_files = {}
        target_files["basket-bag"] = "img_category/bag/basket-bag/*.jpg"
        #target_files["boston-bag"] = "img_category/bag/boston-bag/*.jpg"
        #target_files["backpack"] = "img_category/bag/backpack/*.jpg"
        target_files["hand-bag"] = "img_category/bag/hand-bag/*.jpg"
        #target_files["shoulder-bag"] = "img_category/bag/shoulder-bag/*.jpg"

        #target_files["suitcase"] = "img_category/bag/suitcase/*.jpg"
        #target_files["tote-bag"] = "img_category/bag/tote-bag/*.jpg"
        #target_files["clutch-bag"] = "img_category/bag/clutch-bag/*.jpg"

        return

    def calc_embedding(self):
        features_allcls = self.features_allcls
        X_allcls = np.concatenate([features_allcls[c] for c in features_allcls],axis=0)
        decomp_allcls = decomposition.TruncatedSVD(n_components=2).fit(X_allcls)

        self.comp = decomp_allcls

        Y_allcls = {}
        for cls in features_allcls:
            Y_allcls[cls] = self.comp.transform(self.features_allcls[cls])
        self.comp_features = Y_allcls
        return

    def plot_embedding(self,target_cls=None, output=None, show_thumbnail=True, title=None, thumbnail_size = (32,32), close_thres=4e-3):
        """
        X: PCA出力が入った辞書。辞書のキーは、クラス名
        target_cls: 表示させるクラス。Noneを指定した場合、Xのすべてを表示する
        output: 出力先のフォルダ。Noneの場合、figオブジェクトを返す
        """
        if not self.comp:
            self.calc_embedding()

        X = self.comp_features

        X_allcls = np.concatenate([X[c] for c in X],axis=0)
        x_min, x_max = np.min(X_allcls,axis=0), np.max(X_allcls,axis=0)

        def normalize(p):
            return (p-x_min) / (x_max-x_min)
        
        fig, ax = plt.subplots(1,1,figsize=(10,10),dpi=100)

        idx = -1
        for cls in X:
            idx += 1
            color = self.cmap(idx)
            x = normalize(X[cls])
            ax.scatter(x[:,0],x[:,1],color=color,label=cls,alpha=1.0 if cls==target_cls else 0.3)

        if show_thumbnail and (target_cls is not None):
            shown_images = np.array([[1., 1.]])  # just something big
            x = normalize(X[target_cls])
            for i in range(x.shape[0]):
                dist = np.sum((x[i] - shown_images) ** 2, 1)
                if np.min(dist) < (close_thres * np.min(x_max-x_min)): continue
                shown_images = np.r_[shown_images, [x[i]]]
                img = keras_image.img_to_array(keras_image.load_img(self.fpath_allcls[target_cls][i], target_size=thumbnail_size, interpolation="bicubic"))/255.
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, cmap=plt.cm.gray_r),x[i],bboxprops=dict(edgecolor=self.cmap(list(X.keys()).index(target_cls))))
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        plt.legend()
        if title is not None:
            plt.title(title)
        if output:
            if not os.path.exists(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))
            plt.savefig(output)

        return fig

    def calc_separation_one(self,arr1,arr2):
        x,y = arr1,arr2
        d0 = np.mean(x,axis=0)
        d1 = np.mean(y,axis=0)
        v = d1-d0
        tx = (x-d0).dot(v)/np.power(np.linalg.norm(v),2)
        ty = (y-d0).dot(v)/np.power(np.linalg.norm(v),2)
        d = np.abs(tx.mean() - ty.mean())
        s = np.sqrt(tx.std()**2+ty.std()**2)
        sep = d/s

        return sep,tx,ty,v,d0,d1

    def calc_separation(self,cls1,cls2):
        x = self.X[self.Y==self.cls_list.index(cls1)]
        y = self.X[self.Y==self.cls_list.index(cls2)]
        d0 = np.mean(x,axis=0)
        d1 = np.mean(y,axis=0)
        v = d1-d0
        tx = (x-d0).dot(v)/np.power(np.linalg.norm(v),2)
        ty = (y-d0).dot(v)/np.power(np.linalg.norm(v),2)
        d = np.abs(tx.mean() - ty.mean())
        s = np.sqrt(tx.std()**2+ty.std()**2)
        sep = d/s

        return sep,tx,ty

    def plot_matrix(self,target_cls=None,output=None,mode="hist",doDensity=True,alart_thres = 2.0,title=None):
        assert mode in ["hist","value"]
        features_allcls = self.features_allcls
        cls_list = self.cls_list if not target_cls else target_cls

        fig, axs = plt.subplots(len(cls_list),len(cls_list),figsize=(10, 10))

        for i in range(len(cls_list)):
            for j in range(len(cls_list)):
                ax = axs[i,j]
                ax.set_xticks([])
                ax.set_yticks([])
                if j==0: ax.set_ylabel(cls_list[i])
                if i==len(cls_list)-1: ax.set_xlabel(cls_list[j])      

                if i==j: continue

                sep,tx,ty = self.calc_separation(cls_list[i],cls_list[j])
                
                if mode=="hist":
                    hrange = (min(np.min(tx),np.min(ty)),max(np.max(tx),np.max(ty)))
                    ax.hist(tx,alpha=0.5,range=hrange,density=doDensity)
                    ax.hist(ty,alpha=0.5,range=hrange,density=doDensity)
                elif mode=="value":
                    ax.set_xlim(-1,1)
                    ax.set_ylim(-1,1)
                    ax.set_facecolor(self.cmap(1) if sep<alart_thres else self.cmap(0))
                    ax.text(0, -0.1, "{:.2f}".format(sep),horizontalalignment="center",color="white" if sep<alart_thres else "white")
                else:
                    continue

        if title is not None:
            plt.title(title)
        if output:
            if not os.path.exists(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))
            plt.savefig(output)
        return fig

    def judge_test(self,x,v,d0,d1,max_acc_unnorm_val):
        ta = (x-d0).dot(v)/np.power(np.linalg.norm(v),2)
        return ta<max_acc_unnorm_val

    def judge_one(self,arr1,arr2,nbins=100,thres_recall=0.8,thres_prec=0.8):
        _,t1,t2,v01,d0,d1 = self.calc_separation_one(arr1,arr2)
        hrange = (min(np.min(t1),np.min(t2)),max(np.max(t1),np.max(t2)))
        h1,xaxis = np.histogram(t1,bins=nbins,range=hrange,density=False)
        h2,_ = np.histogram(t2,bins=nbins,range=hrange,density=False)
        An, Au, Rn, Pn, Ru, Pu = [np.zeros((nbins,),dtype=np.float32) for _ in range(6)]
        s1 = h1.sum() / (h1.sum()+h2.sum())
        s2 = h2.sum() / (h1.sum()+h2.sum())
        for i in range(nbins):
            g1 = h1[:i].sum()
            g2 = h2[i:].sum()
            b1 = h1[i:].sum()
            b2 = h2[:i].sum()

            #print(i,g1,g2,b1,b2)
            Au[i] = (g1+g2)/(g1+g2+b1+b2)
            Ru[i] = g1/(g1+b1)
            Pu[i] = g1/(g1+b2)

            g1 *= s1
            g2 *= s2
            b1 *= s1
            b2 *= s2

            An[i] = (g1+g2)/(g1+g2+b1+b2)
            #print(g1,g2,b1,b2)
            Rn[i] = g1/(g1+b1)
            Pn[i] = g1/(g1+b2)

        Mu = np.argmax(Au) # 学習時の最適スレッショルド
        Vu = xaxis[:-1][Mu] # 上記と対応する数値

        mRu,mRn = Ru[Mu],Rn[Mu]
        mPu,mPn = Pu[Mu],Pn[Mu]

        NGexample1 = np.where(t1>Vu)[0]
        NGexample1 = NGexample1[np.argsort(t1[NGexample1])][::-1] # 降順にする
        NGexample2 = np.where(t2<Vu)[0]
        NGexample2 = NGexample2[np.argsort(t2[NGexample2])][::+1] # 昇順にする

        res = {}
        res["mean_vector_0"] = d0
        res["mean_vector_1"] = d1
        res["mean_vector_0_to_1"] = v01
        res["max_acc_unnorm_thres"] = Mu
        res["max_acc_unnorm_val"] = Au[Mu]
        res["max_recall_unnorm_val"] = mRu
        res["max_prec_unnorm_val"] = mPu
        res["max_recall_norm_val"] = mRn
        res["max_prec_norm_val"] = mPn
        res["recall_norm"] = Rn
        res["prec_norm"] = Pn
        res["acc_norm"] = An
        res["recall_unnorm"] = Ru
        res["prec_unnorm"] = Pu
        res["acc_unnorm"] = Au
        res["xaxis"] = xaxis[:-1]
        res["NG_recall_example_index"] = NGexample1
        res["NG_prec_example_index"] = NGexample2

        # 分類
        isOK_unnorm = (res["max_recall_unnorm_val"]>thres_recall) and (res["max_prec_unnorm_val"]>thres_prec)
        isOK_norm   = (res["max_recall_norm_val"]  >thres_recall) and (res["max_prec_norm_val"]  >thres_prec)

        flag = -1
        if s1==0 or s2==0: flag = -1 # 片方の分布が空
        elif isOK_unnorm and isOK_norm: flag = 1 # 十分に分布が分離している
        elif isOK_norm: flag = 2 # データ数が少ない
        else: # ノーマライズしたときにもそもそも分布が重なっている
            recall = res["max_recall_norm_val"]
            prec   = res["max_prec_norm_val"]
            if (recall < thres_recall) and (prec < thres_prec): flag = 3 # 分布が完全に重なってしまっている
            elif recall < thres_recall: flag = 4 # 自分のカテゴリの中に、相手に似ているものがある
            elif prec   < thres_prec  : flag = 5 # 相手のカテゴリの中に、自分に似ているものがある

        res["judgement"] = flag

        return res

    """
    def get_gradcam(self,fig,img,v,d1):
        img_orig, cam = self.grad_cam(img,v,d1)
        dpi = 80.0
        xpixels, ypixels = img_orig.size
        #plt.ioff()
        fig = plt.figure(figsize=(xpixels/dpi, ypixels/dpi),dpi=dpi)

        plt.imshow(img_orig)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')

        f = io.BytesIO()
        plt.savefig(f,format="png",pad_inches=0,dpi=dpi)
        f.flush()
        f.seek(0)
        img = Image.open(f)
        return img
    """


if __name__=="__main__":
    rd = rapidDiag()
    rd.set_model()
    rd.set_files()
    rd.process()
    rd.plot_embedding(target_cls=None,output="output/emb_all.png")
    #rd.plot_embedding(target_cls="basket-bag",output="output/emb_basket-bag.png")
    for cls in rd.cls_list:
        rd.plot_embedding(target_cls=cls,output="output/emb_{}.png".format(cls))

    rd.plot_matrix(target_cls=None,output="output/mat_hist.png",mode="hist",doDensity=True,title=None)
    rd.plot_matrix(target_cls=None,output="output/mat_value.png",mode="value",alart_thres = 2.0,title=None)
