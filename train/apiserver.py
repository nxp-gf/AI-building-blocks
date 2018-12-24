#!/usr/bin/env python
from flask import Flask, send_from_directory
from flask_restful import reqparse, abort, Api, Resource, request
import os

import sys

app = Flask(__name__)
api = Api(app)

class TrainModels(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('name', type=str, location='args')
        self.modelspath = "./models"
    def get(self):
        return send_from_directory(self.modelspath, data.name, as_attachment=True)

    def delete(self):
        return {'state':'SUCCESS'}, 200

    def put(self):
        return {'state':'SUCCESS'}, 200

    def post(self):
        return {'state':'SUCCESS'}, 200

##
## Actually setup the Api resource routing here
##
api.add_resource(TrainModels, '/models')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, threaded=True)

