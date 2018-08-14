from flask import Flask,session, redirect, url_for, escape, request, render_template, jsonify
import sqlite3
from src.rapidDiag import rapidDiag
import io
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import base64

app = Flask(__name__)
app.debug = True
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

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

# Converts TEXT to np.array when selecting
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

@app.route('/')
def index():
    # TODO: ログイン対応
    return render_template('index.html', user_name="test user")

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
        project_name = request.args.get('project_name', type=str)
    except:
        return jsonify({"status":"existed"})
    
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)

    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()
    # 上書き確認
    cur.execute("SELECT * from ProjectDB where CategoryDB_name=?",
                (CategoryDB_name,))
    if not len(cur.fetchall())==0:
        return jsonify({"status":"existed"})

    cur.execute("INSERT into ProjectDB values (?,?,?)",
                (user_name,project_name,CategoryDB_name))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/del_project',methods=["POST"])
def del_project():
    try:
        project_name = request.args.get('project_name', type=str)
    except:
        return jsonify({"status":"existed"})
    
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)

    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()
    # 存在確認
    cur.execute("SELECT * from ProjectDB where CategoryDB_name=?",
                (CategoryDB_name,))
    if len(cur.fetchall())==0:
        return jsonify({"status":"not existed"})

    cur.execute("DELETE from ProjectDB where CategoryDB_name=?",
                (CategoryDB_name,))
    con.commit()
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
    cur.execute("create table IF NOT EXISTS DataDB__{} (category_id integer, filename text, feature arr)".format(CategoryDB_name))
    con.commit()

    cur.execute("INSERT into DataDB__{} values (?,?,?)".format(CategoryDB_name), 
                (category_id,img_filename, feature))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/change_category',methods=["POST"])
def change_category():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        project_name  = request.args.get('project_name', type=str)
        category_id   = request.args.get('category_id', type=int)
        category_name = request.args.get('category_name', type=str)
        category_type = request.args.get('category_type', type=str)
    except:
        return jsonify({"status":"fail"})
    if not category_type in ["train","test"]: return jsonify({"status":"incorrect type:".format(category_type)})

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)
    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()
    cur.execute("create table IF NOT EXISTS CategoryDB__{} (category_id integer primary key, category_name text, category_type text)".format(CategoryDB_name))
    con.commit()

    cur.execute("INSERT OR REPLACE into CategoryDB__{} values (?,?,?)".format(CategoryDB_name), 
                (category_id, category_name, category_type))
    con.commit()

    return jsonify({"status":"success"})

@app.route('/get_category',methods=["POST"])
def get_category():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        project_name  = request.args.get('project_name', type=str)
    except:
        return jsonify({"status":"fail"})

    # DBを更新
    CategoryDB_name = generate_CategoryDB_name(user_name,project_name)
    con = sqlite3.connect(DBFilePath)
    cur = con.cursor()

    try:
        cur.execute("SELECT * from CategoryDB__{}".format(CategoryDB_name))
        res = {"count":0,"items":[]}
        for line in cur:
            res["count"] += 1
            res["items"].append({"category_id":line[0],"category_name":line[1],"category_type":line[2]})
        return jsonify(res)
    except:
        res = {"count":0,"items":[]}
        return jsonify(res)

@app.route('/delete_category',methods=["POST"])
def delete_category():
    user_name = "test user" # TODO: sessionができたら、これもログイン名に直す
    try:
        project_name  = request.args.get('project_name', type=str)
        category_id   = request.args.get('category_id', type=int)
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
if __name__ == '__main__':
    app.run()
