from flask import Flask,session, redirect, url_for, escape, request, render_template, jsonify, helpers
from functools import wraps
from flask_login import login_user, logout_user, LoginManager, UserMixin
import sqlite3
import matplotlib as mpl
mpl.use('Agg')
from src.rapidDiag import rapidDiag
import io
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import uuid

app = Flask(__name__)
app.debug = True
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "users.login" # login_viewのrouteを設定

# TODO: すべての関数に、認証を追加

# 初期化

#### Sqlite3にnumpy arrayを収納できるようにする
def adapt_array(arr):
    #http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting -> どうも自動で動いてくれていない・・・
sqlite3.register_converter("array", convert_array)
## TODO:UserDB

DBFilePath = "db.sqlite"

## ProjectDB
con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
cur = con.cursor()
cur.execute("create table IF NOT EXISTS ProjectDB (user_name text, project_name text, CategoryDB_name text)")
con.commit()

## Deep Learningモデル
rds = rapidDiag(platform="server")
rds.set_model(model_type="mobilenet",model_layer="last")

#################################################################################
@app.route('/login', methods=['GET', 'POST'])
def login():
    # ToDo: パスワードチェックをかける
    if request.method == 'POST':
        session['user_name'] = request.form['user_name']
        return redirect(url_for('index'))
    return '''
        <form method="post">
            <p><input type=text name=user_name>
            <p><input type=submit value=Login>
        </form>
    '''

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not 'user_name' in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/logout')
def logout():
    # remove the user_name from the session if it's there
    session.pop('user_name', None)
    return redirect(url_for('index'))

#################################################################################
@app.route('/')
@login_required
def index():
    #session['project_name'] = uuid.uuid4() # セッションごとに別のプロジェクトを作れる
    session['project_name'] = session['user_name'] # セッションを跨いで結果が保存される
    #new_project()
    return render_template('index.html', user_name=session['user_name'], project_name = session['project_name'], css_url = url_for('static', filename='style.css'))

#def generate_CategoryDB_name(user_name,project_name):
def generate_CategoryDB_name():
    user_name = session['user_name']
    project_name = session['project_name']
    CategoryDB_name  = user_name.replace(" ","_").replace("/","_").replace(".","_")
    CategoryDB_name += "__"
    CategoryDB_name += project_name.replace(" ","_").replace("/","_").replace(".","_")
    return CategoryDB_name

#################################################################################

@app.route('/new_project',methods=["POST"])
@login_required
def new_project():

    # すでに存在している場合には、まず削除
    del_project()

    # 新たに作る
    CategoryDB_name = generate_CategoryDB_name()

    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()
    # 上書き確認
    cur.execute("SELECT * from ProjectDB where CategoryDB_name=?",
                (CategoryDB_name,))
    if not len(cur.fetchall())==0:
        return jsonify({"status":"existed"})

    # CategoryDB__XXを作成
    cur.execute("create table IF NOT EXISTS CategoryDB__{} (category_id integer primary key, category_type text, category_name text, center array)".format(CategoryDB_name))
    con.commit()

    # DataDB__XXを作成
    cur.execute("create table IF NOT EXISTS DataDB__{} (category_id integer, data_id integer, filename text, feature array, image blob, primary key(category_id,data_id))".format(CategoryDB_name))
    con.commit()

    # ProjectDBに記録
    user_name = session['user_name']
    project_name = session['project_name']
    cur.execute("INSERT into ProjectDB values (?,?,?)",
                (user_name,project_name,CategoryDB_name))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/del_project',methods=["POST"])
@login_required
def del_project():
    CategoryDB_name = generate_CategoryDB_name()

    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()

    ################
    # ProjectDBから削除
    cur.execute("SELECT * from ProjectDB where CategoryDB_name=?",
                (CategoryDB_name,))
    if len(cur.fetchall())>0:
        print("delete column from ProjectDB")
        cur.execute("DELETE from ProjectDB where CategoryDB_name=?",
                    (CategoryDB_name,))
        con.commit()

    ################
    # CategoryDB__XX, DataDB__XXを削除
    try:
        cur.execute("DROP TABLE CategoryDB__{}".format(CategoryDB_name))
        con.commit()
        print("delete CategoryDB__{}".format(CategoryDB_name))
    except:
        pass

    ##
    try:
        cur.execute("DROP TABLE DataDB__{}".format(CategoryDB_name))
        con.commit()
        print("delete DataDB__{}".format(CategoryDB_name))
    except:
        pass

    return jsonify({"status":"success"})

#################################################################################
@app.route('/upload/<int:category_id>',methods=["POST"])
@login_required
def upload(category_id):
    try:
        img_data = request.files["file"]
        img_filename = img_data.filename
        img = Image.open(BytesIO(img_data.read()))
        img_resized = img.resize(rds.image_size[:2],Image.BICUBIC)
        feature = rds.process_one(img)
    except:
        return jsonify({"status":"incorrect image file"})

    # DBへ保存
    CategoryDB_name = generate_CategoryDB_name()
    con = sqlite3.connect(DBFilePath, detect_types=sqlite3.PARSE_DECLTYPES,isolation_level="IMMEDIATE",timeout=60*1000) # ここは次々とリクエストが来るとユニーク性が失われてしまうので、IMMEDIATEにしておく必要
    cur = con.cursor()
    cur.execute("SELECT COUNT (*) FROM DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                    (category_id,))
    items_uploaded = cur.fetchone()[0]
    data_id = items_uploaded

    stream = io.BytesIO()
    img_resized.save(stream, format="JPEG")
    stream.seek(0)

    # ロックが発生するので修正必要
    cur.execute("INSERT into DataDB__{} values (?,?,?,?,?)".format(CategoryDB_name), 
                (category_id,data_id,img_filename, feature, stream.getvalue()))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/add_category',methods=["POST"])
@login_required
def add_category():

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name()
    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()

    # そもそも、きちんとプロジェクトが作成されているか？
    cur.execute("SELECT count(*) from sqlite_master where type='table' and name='CategoryDB__{}'".format(CategoryDB_name));
    if cur.fetchone()[0]==0: new_project()
    cur.execute("SELECT count(*) from sqlite_master where type='table' and name='DataDB__{}'".format(CategoryDB_name));
    if cur.fetchone()[0]==0: new_project()

    # カテゴリを追加
    cur.execute("SELECT * from CategoryDB__{}".format(CategoryDB_name))
    max_idx = -1
    for line in cur:
        category_id = line[0]
        max_idx = max(max_idx,category_id)
    next_idx = max_idx+1

    # レコードを作成
    cur.execute("INSERT into CategoryDB__{} values (?,?,?,?)".format(CategoryDB_name), 
                (next_idx, "Category{}".format(next_idx), "train", np.zeros(rds.output_dim,)))
    con.commit()

    return jsonify({"status":"success","category_id":next_idx})

@app.route('/change_category',methods=["POST"])
@login_required
def change_category():
    try:
        category_id   = request.json['category_id']
        category_name = request.json['category_name']
        category_type = request.json['category_type']
        print(category_id,category_name,category_type)
    except:
        return jsonify({"status":"fail"})
    if not category_type in ["train","test"]: return jsonify({"status":"incorrect type:".format(category_type)})

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name()
    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()

    # TODO:Updateのみに修正
    cur.execute("REPLACE into CategoryDB__{} values (?,?,?,?)".format(CategoryDB_name), 
                (category_id, category_name, category_type, np.zeros(rds.output_dim,)))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/get_category',methods=["POST"])
@login_required
def get_category():
    # DBを更新
    CategoryDB_name = generate_CategoryDB_name()
    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()

    # そもそも、きちんとプロジェクトが作成されているか？
    cur.execute("SELECT count(*) from sqlite_master where type='table' and name='CategoryDB__{}'".format(CategoryDB_name));
    if cur.fetchone()[0]==0: new_project()
    cur.execute("SELECT count(*) from sqlite_master where type='table' and name='DataDB__{}'".format(CategoryDB_name));
    if cur.fetchone()[0]==0: new_project()

    cur.execute("SELECT * from CategoryDB__{}".format(CategoryDB_name))
    res = {"count":0,"items":[]}
    for line in cur.fetchall():
        category_id = line[0]
        cur.execute("SELECT COUNT (*) FROM DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                    (category_id,))
        items_uploaded = cur.fetchall()[0][0]
        res["count"] += 1
        res["items"].append({"category_id":line[0],"category_name":line[1],"category_type":line[2],"items_uploaded":items_uploaded})
    return jsonify(res)

@app.route('/delete_category',methods=["POST"])
@login_required
def delete_category():
    category_id  = request.json['category_id']

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name()
    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()

    # まず、CategoryDBを更新
    cur.execute("DELETE from CategoryDB__{} where category_id=?".format(CategoryDB_name),
                (category_id,))
    con.commit()

    # もし、DataDBにデータがあるのならば、それらも削除
    cur.execute("DELETE FROM DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                (category_id,))
    con.commit()

    return jsonify({"status":"success"})

#################################################################################
@app.route('/evaluate',methods=["POST"])
@login_required
def evaluate():

    # trainのカテゴリを抽出する
    CategoryDB_name = generate_CategoryDB_name()
    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()

    cur.execute("SELECT * from CategoryDB__{}".format(CategoryDB_name))
    allItems = cur.fetchall()
    category_name = {x[0]:x[1] for x in allItems}
    category_type = {x[0]:x[2] for x in allItems}
    category_id_train = [x[0] for x in allItems if x[2]=="train"]
    category_id_test  = [x[0] for x in allItems if x[2]=="test"]
    category_id_correspondence = {}
    for k in category_id_train:
        name = category_name[k]
        category_id_correspondence[k] = -1
        for m in category_id_test:
            if category_name[m] == name:
                category_id_correspondence[k] = m
                break

    result = {"train_pairs":[],
              "train_matrix":{
                  "size": len(category_id_train),
                  "col": [category_name[x] for x in category_id_train],
                  "idx": category_id_train}
              }
    for c1 in category_id_train:
        # DBから情報を抽出する
        cur.execute("SELECT * from DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                (c1,))
        f1 = cur.fetchall()
        arr1 = np.array([convert_array(x[3])[0] for x in f1])
        did1 = [x[1] for x in f1]

        # DBへ書き込む
        CategoryDB_name = generate_CategoryDB_name()
        con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
        cur = con.cursor()
        cur.execute("REPLACE into CategoryDB__{} values (?,?,?,?)".format(CategoryDB_name), 
                    (c1, category_name[c1], category_type[c1], np.mean(arr1,axis=0)))
        con.commit()

        test_items = []

        for c2 in category_id_train:
            if c1==c2: continue
            # DBから情報を抽出する
            cur.execute("SELECT * from DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                    (c2,))
            f2 = cur.fetchall()
            arr2 = np.array([convert_array(x[3])[0] for x in f2])
            did2 = [x[1] for x in f2]
            #exp2 = [x[1] for x in f2]
            item = {}
            item["category_id_1"] = c1
            item["category_id_2"] = c2
            item["category_name_1"] = category_name[c1]
            item["category_name_2"] = category_name[c2]
            jRes = rds.judge_one(arr1,arr2,nbins=100,thres_recall=0.8,thres_prec=0.8)
            item["result_flag"] =flag= jRes["judgement"]
            item["result_example1"] = []
            item["result_example2"] = []

            test_items.append({"c2":c2,"max_acc_unnorm_val":jRes["max_acc_unnorm_val"],"mean_vector_0":jRes["mean_vector_0"],"mean_vector_1":jRes["mean_vector_1"],"mean_vector_0_to_1":jRes["mean_vector_0_to_1"],"judgement":jRes["judgement"]})

            #if flag in [3,4,5]: # 事例をいれていく
            example_did1 = [(c1,did1[i]) for i in jRes["NG_recall_example_index"].tolist()]
            example_did2 = [(c2,did2[i]) for i in jRes["NG_prec_example_index"].tolist()]
            #example_did = []
            if flag==0 or flag==1: # 正解データを入れていく
                item["result_example1"] = [(c1,x) for x in did1]
                item["result_example2"] = []
            elif flag==3: # 互い違いに入れていく
                item["result_example1"] = example_did1
                item["result_example2"] = example_did2
            elif flag==4:
                item["result_example1"] = example_did1
                item["result_example2"] = []
                #example_did=example_did1
            elif flag==5:
                item["result_example1"] = []
                item["result_example2"] = example_did2


            result["train_pairs"].append(item)

        # もし評価用データが存在する場合は、それも評価する
        c1_test = category_id_correspondence[c1]
        if c1_test>=0:
            cur.execute("SELECT * from DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                    (c1_test,))
            f1_test = cur.fetchall()
            arr1_test = np.array([convert_array(x[3])[0] for x in f1_test])

            rflg = np.ones((arr1_test.shape[0],),dtype=np.bool)
            rflg[:] = True

            for item in test_items:
                if not item["judgement"] == 1: continue # そもそも、あたっているものでないと適用してもしょうがない
                f = rds.judge_test(arr1_test,
                                   v=item["mean_vector_0_to_1"],
                                   d0=item["mean_vector_0"],
                                   d1=item["mean_vector_1"],
                                   max_acc_unnorm_val=item["max_acc_unnorm_val"])

                rflg = np.logical_and(rflg,f)

            f_good = np.sum(rflg) /  float(rflg.shape[0])
            is_good = f_good > 0.6 # 0.6以上が入っていることを基準とする

            for r in result["train_pairs"]:
                if not r["category_id_1"]==c1:
                    continue

                if r["result_flag"] == 1: # とりあえず学習可能なサンプル
                    if is_good: # 評価データも良いと言っている
                        r["result_flag"] = 0 # 最高評価
                    else: # 学習の結果は良いが、評価では性能がでない様子
                        r["result_flag"] = 6

    print(result)
    result["status"] = "success"

    return jsonify(result)

#################################################################################
@app.route('/show_image/<int:category_id>/<int:data_id>',methods=["GET"])
@login_required
def show_image(category_id,data_id):
    CategoryDB_name = generate_CategoryDB_name()
    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()
    cur.execute("SELECT * from DataDB__{} WHERE category_id=? and data_id=?".format(CategoryDB_name),
            (category_id,data_id))
    f = cur.fetchone()
    img_file = io.BytesIO(f[4])
    img = Image.open(img_file)
    img_file_to_send = io.BytesIO()
    img.save(img_file_to_send,format="png")

    response = helpers.make_response(img_file_to_send.getvalue())
    response.headers["Content-type"] = "Image"
    return response
#################################################################################
@app.route('/show_gradcam/<int:category_id_1>/<int:category_id_2>/<int:data_id_1>',methods=["GET"])
@login_required
def gradcam(category_id_1,category_id_2,data_id_1):
    # Step1: カテゴリの中心を抽出
    CategoryDB_name = generate_CategoryDB_name()
    con = sqlite3.connect(DBFilePath,isolation_level="DEFERRED",timeout=60*1000)
    cur = con.cursor()

    cur.execute("SELECT * from CategoryDB__{} WHERE category_id=?".format(CategoryDB_name),
               (category_id_1,))
    category_center_1 = convert_array(cur.fetchone()[3])

    cur.execute("SELECT * from CategoryDB__{} WHERE category_id=?".format(CategoryDB_name),
               (category_id_2,))
    category_center_2 = convert_array(cur.fetchone()[3])

    category_diff = category_center_2 - category_center_1

    # Step2: imgを読み込み
    cur.execute("SELECT * from DataDB__{} WHERE category_id=? and data_id=?".format(CategoryDB_name),
               (category_id_1,data_id_1))
    img_resized_file = io.BytesIO(cur.fetchone()[4])
    img_resized = Image.open(img_resized_file)

    # Step3: grad-camを実行
    img_orig,cam = rds.grad_cam(img_resized,category_diff, category_center_2) # 2を入れることに注意
    heatmap_img = cv2.resize(cam, rds.image_size[:2][::-1])
    cam *= 255.
    cam = cam.astype(np.uint8)
    heatmap_img = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img = cv2.addWeighted(heatmap_img, 0.5, np.asarray(img_orig), 0.5, 0)
    img = Image.fromarray(img[:,:,::-1])

    img_file_to_send = io.BytesIO()
    img.save(img_file_to_send,format="png")

    response = helpers.make_response(img_file_to_send.getvalue())
    response.headers["Content-type"] = "Image"
    return response

#################################################################################
if __name__ == '__main__':
    #app.run()
    app.run(debug=False, host='0.0.0.0', port=80)
