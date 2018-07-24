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

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class DataReciever(tornado.web.RequestHandler):
    # https://github.com/zhanglongqi/TornadoAJAXSample
    def get(self):
        example_response = {}
        example_response['name'] = 'example'
        example_response['width'] = 1020

        self.write(json.dumps(example_response))

    def post(self):
        json_obj = json_decode(self.request.body)
        print('Post data received')

        for key in list(json_obj.keys()):
            print('key: %s , value: %s' % (key, json_obj[key]))

        # new dictionary
        response_to_send = {}
        response_to_send['newkey'] = json_obj['key1']

        print('Response to return')

        pprint.pprint(response_to_send)

        self.write(json.dumps(response_to_send))

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
                (r"/", MainHandler),
                (r"/test/", DataReciever)
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
