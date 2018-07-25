import os
import logging
import tornado.auth
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import os.path
import uuid
import json
import pprint
from  tornado.escape import json_decode
from  tornado.escape import json_encode
from tornado.options import define, options

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from src.rapidDiag import rapidDiag

import sqlite3
import numpy as np
import io


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


# TODO: 基本的な画面を表示するAPI
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

# TODO: データを受信して、処理する
class DataReciever(tornado.web.RequestHandler):
    def initialize(self):
        self.rds = {}
        self.rds["deep"] = rapidDiag(platform="server")
        self.rds["deep"].set_model(model_type="mobilenet",model_layer="last")

        self.con = con = sqlite3.connect("temp/db1.sqlite", detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = cur = con.cursor()
        cur.execute("create table IF NOT EXISTS test (string filename, integer category, arr feature)")
        con.commit()

        return

    # https://github.com/zhanglongqi/TornadoAJAXSample
    def get(self):
        example_response = {}
        self.write(json.dumps(example_response))
        return

    def post(self,category):
        fileinfo = self.request.files['file'][0]
        fileName = fileinfo['filename']
        print(category,fileinfo["filename"])
        img = Image.open(BytesIO(fileinfo["body"]))
        feature = self.rds["deep"].process_one(img)
        
        self.cur.execute("insert into test values (?,?,?)", (fileName, category, feature))
        self.con.commit()

        # レスポンスを返す
        response_to_send = {}
        self.write(json.dumps(response_to_send))
        return

# TODO: データの名前・データ種別を設定する

# TODO: データの名前・データ種別を設定する

# TODO: データの分離レベルを取得する

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
                (r"/", MainHandler),
                (r"/upload/(?P<category>\d+)", DataReciever)
            ]
        # settingsでdebugをTrueに設定。設定はこれだけ。
        settings = dict(
            debug=True,
            template_path=os.path.join(os.getcwd(),  "templates"),
            static_path=os.path.join(os.getcwd(),  "static"),
            )
        tornado.web.Application.__init__(self,handlers,**settings)


if __name__ == "__main__":
    application = tornado.httpserver.HTTPServer(Application())
    application.listen(9000)
    print("Server is up ...")
    tornado.ioloop.IOLoop.instance().start()
