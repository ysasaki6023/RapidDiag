from flask import Flask,session, redirect, url_for, escape, request, render_template, jsonify
from flask_login import login_user, logout_user, LoginManager, UserMixin
import sqlite3
from src.rapidDiag import rapidDiag
import io
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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

DBFilePath = "temp/db.sqlite"
## ProjectDB
con = sqlite3.connect(DBFilePath)
cur = con.cursor()
cur.execute("create table IF NOT EXISTS ProjectDB (user_name text, project_name text, CategoryDB_name text)")
con.commit()

## Deep Learningモデル
rds = rapidDiag(platform="server")
rds.set_model(model_type="mobilenet",model_layer="last")

#################################################################################
@app.route('/login')
def login():
    if valid():
        login_user(user)
        return redirect(request.args.get('next') or url_for('index'))

@app.route('/logout')
def logout():
    logout_user()

#################################################################################
@app.route('/')
def index():
    # TODO: ログイン対応
    return render_template('index.html', user_name="test user", css_url = url_for('static', filename='style.css'))

def generate_CategoryDB_name(user_name,project_name):
    CategoryDB_name  = user_name.replace(" ","_").replace("/","_").replace(".","_")
    CategoryDB_name += "__"
    CategoryDB_name += project_name.replace(" ","_").replace("/","_").replace(".","_")
    return CategoryDB_name

#################################################################################
@app.route('/show_project',methods=["POST","GET"])
def show_project():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()
    cur.execute("SELECT * from ProjectDB where user_name=?",
                 (user_name,))


    jstr = {"count":0,"project_name":[]}
    for row in cur:
        jstr["count"] += 1
        jstr["project_name"].append(row[1])

    return jsonify(jstr)

@app.route('/new_project',methods=["POST"])
def new_project():
    try:
        project_name = request.json["project_name"]
    except:
        return jsonify({"status":"failed"})
    
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)

    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()
    # 上書き確認
    cur.execute("SELECT * from ProjectDB where CategoryDB_name=?",
                (CategoryDB_name,))
    if not len(cur.fetchall())==0:
        return jsonify({"status":"existed"})

    # ProjectDBに記録
    cur.execute("INSERT into ProjectDB values (?,?,?)",
                (user_name,project_name,CategoryDB_name))
    con.commit()

    # CategoryDB__XXを作成
    cur.execute("create table IF NOT EXISTS CategoryDB__{} (category_id integer primary key, category_name text, category_type text)".format(CategoryDB_name))
    con.commit()

    # DataDB__XXを作成
    cur.execute("create table IF NOT EXISTS DataDB__{} (category_id integer, filename text, feature array)".format(CategoryDB_name))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/del_project',methods=["POST"])
def del_project():
    try:
        project_name = request.json["project_name"]
    except:
        return jsonify({"status":"existed"})
    
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)

    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()

    ################
    # ProjectDBから削除
    cur.execute("SELECT * from ProjectDB where CategoryDB_name=?",
                (CategoryDB_name,))
    if len(cur.fetchall())>0:
        cur.execute("DELETE from ProjectDB where CategoryDB_name=?",
                    (CategoryDB_name,))
        con.commit()

    ################
    # CategoryDB__XX, DataDB__XXを削除
    try:
        cur.execute("DROP TABLE CategoryDB_name=?",
                    (CategoryDB_name,))
        con.commit()
    except:
        pass
    ##
    try:
        cur.execute("DROP TABLE DataDB_name=?",
                    (CategoryDB_name,))
        con.commit()
    except:
        pass

    return jsonify({"status":"success"})

#################################################################################
@app.route('/upload/<project_name>/<int:category_id>',methods=["POST"])
def upload(project_name,category_id):
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        img_data = request.files["file"]
        img_filename = img_data.filename
        img = Image.open(BytesIO(img_data.read()))
        feature = rds.process_one(img)
    except:
        return jsonify({"status":"incorrect image file"})

    # DBへ保存
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)
    con = sqlite3.connect(DBFilePath, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    cur.execute("INSERT into DataDB__{} values (?,?,?)".format(CategoryDB_name), 
                (category_id,img_filename, feature))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/add_category',methods=["POST"])
def add_category():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        project_name = request.json["project_name"]
        print(request.json)
    except:
        return jsonify({"status":"fail"})

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)
    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()
    cur.execute("SELECT * from CategoryDB__{}".format(CategoryDB_name))
    max_idx = -1
    for line in cur:
        category_id = line[0]
        max_idx = max(max_idx,category_id)
    next_idx = max_idx+1

    # レコードを作成
    cur.execute("INSERT into CategoryDB__{} values (?,?,?)".format(CategoryDB_name), 
                (next_idx, "temp_name", "train"))
    con.commit()

    return jsonify({"status":"success","category_id":next_idx})

@app.route('/change_category',methods=["POST"])
def change_category():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        project_name  = request.json['project_name']
        category_id   = request.json['category_id']
        category_name = request.json['category_name']
        category_type = request.json['category_type']
    except:
        return jsonify({"status":"fail"})
    if not category_type in ["train","test"]: return jsonify({"status":"incorrect type:".format(category_type)})

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)
    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()

    # TODO:Updateのみに修正
    cur.execute("INSERT OR REPLACE into CategoryDB__{} values (?,?,?)".format(CategoryDB_name), 
                (category_id, category_name, category_type))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/get_category',methods=["POST"])
def get_category():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        project_name  = request.json['project_name']
        print(request.json)
    except:
        return jsonify({"status":"fail"})

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)
    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()

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
    #except:
    #    res = {"count":0,"items":[]}
    #    return jsonify(res)

@app.route('/delete_category',methods=["POST"])
def delete_category():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        project_name  = request.json['project_name']
        category_id   = request.json['category_id']
    except:
        return jsonify({"status":"fail"})

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)
    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()

    # まず、CategoryDBを更新
    try:
        cur.execute("DELETE from CategoryDB__{} where category_id=?".format(CategoryDB_name),
                    (category_id,))
        con.commit()
    except:
        # Tableが存在しなかった
        pass

    # もし、DataDBにデータがあるのならば、それらも削除
    try:
        cur.execute("DELETE FROM DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                    (category_id,))
        con.commit()
    except:
        # Tableが存在しなかった
        pass

    return jsonify({"status":"success"})

#################################################################################
@app.route('/evaluate',methods=["POST"])
def evaluate():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        project_name  = request.json['project_name']
    except:
        return jsonify({"status":"fail"})

    # trainのカテゴリを抽出する
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)
    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()

    cur.execute("SELECT * from CategoryDB__{}".format(CategoryDB_name))
    allItems = cur.fetchall()
    category_name = {x[0]:x[1] for x in allItems}
    category_id_train = [x[0] for x in allItems if x[2]=="train"]

    result = {"train_pairs":[],
              "train_matrix":{
                  "size": len(category_id_train),
                  "col": [category_name[x] for x in category_id_train]
             }}
    for c1 in category_id_train:
        # DBから情報を抽出する
        cur.execute("SELECT * from DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                (c1,))
        f1 = cur.fetchall()
        arr1 = np.array([convert_array(x[2])[0] for x in f1])
        exp1 = [x[1] for x in f1]
        for c2 in category_id_train:
            if c1==c2: continue
            cur.execute("SELECT * from DataDB__{} WHERE category_id=?".format(CategoryDB_name),
                    (c2,))
            f2 = cur.fetchall()
            arr2 = np.array([convert_array(x[2])[0] for x in f2])
            exp2 = [x[1] for x in f2]
            # DBから情報を抽出する
            item = {}
            item["category_id_1"] = c1
            item["category_id_2"] = c2
            item["category_name_1"] = category_name[c1]
            item["category_name_2"] = category_name[c2]
            jRes = rds.judge_one(arr1,arr2,nbins=100,thres_recall=0.8,thres_prec=0.8)
            item["result_flag"] =flag= jRes["judgement"]
            item["result_example"] = []

            if flag in [3,4,5]: # 事例をいれていく
                example_url1 = [exp1[i] for i in jRes["NG_recall_example_index"].tolist()]
                example_url2 = [exp2[i] for i in jRes["NG_prec_example_index"].tolist()]
                example_url = []
                if   flag==3: # 互い違いに入れていく
                    minIdx = min(len(example_url1),len(example_url2))
                    for i in range(minIdx):
                        example_url.append(example_url1[i])
                        example_url.append(example_url2[i])
                    example_url += example_url1[minIdx:]
                    example_url += example_url2[minIdx:]
                elif flag==4: example_url=example_url1
                elif flag==5: example_url=example_url2

                item["result_example"] = example_url
            result["train_pairs"].append(item)
    print(result)
    result["status"] = "success"

    return jsonify(result)

#################################################################################
if __name__ == '__main__':
    app.run()
