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

# TODO: 基本的な画面を表示するAPI
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

# TODO: データを受信して、処理する
class DataReciever(tornado.web.RequestHandler):
    # https://github.com/zhanglongqi/TornadoAJAXSample
    def get(self):
        example_response = {}
        example_response['name'] = 'example'
        example_response['width'] = 1020
        self.write(json.dumps(example_response))
        return

    def post(self):
        fileinfo = self.request.files['file'][0]
        fileName = fileinfo['filename']
        print(fileinfo["filename"])
        img = Image.open(BytesIO(fileinfo["body"]))

        # TODO: 画像をNNにより分析する

        # TODO: 分析結果をDBへ保存する

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
                (r"/upload", DataReciever)
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
