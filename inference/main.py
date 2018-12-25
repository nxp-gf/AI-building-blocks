#!/usr/bin/env python
from importlib import import_module
import os,time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO,emit
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import time,requests
import multiprocessing

app = Flask(__name__)
socketio = SocketIO()
socketio.init_app(app)

MODEL_NAME = './models/newmodel'

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index_web.html')

def download_models(name):
    print "Downloading modules from cloud."
    svrurl  = 'http://10.193.20.190:5050/models?name=' + name
    headers = {"Content-type":"application/json","Accept": "application/json"}
    r = requests.get(svrurl, headers=headers)
    if (r.status_code == 200):
        f = open(MODEL_NAME, 'w')
        f.write(r.content)
        f.close()

        print "Modules updated successfully"
        return True
    else:
        print "Modules updated failed"
        return False


def kafka_handler(needupdate):
    from kafka import KafkaConsumer

    topic_updated  = "NXP_MODELS_UPDATED"
    consumer = KafkaConsumer(bootstrap_servers='10.193.20.98:9092')
    consumer.subscribe([topic_updated])
    for msg in consumer:
        print(msg.value)
        download_models(msg.value)
        needupdate.value = 1

def recognition_handler(num, result, needupdate):
    automodel = load_model('models/model.h4')
    while(True):
        if (needupdate.value != 0):
            automodel = load_model(MODEL_NAME)
            needupdate.value = 0
        if (num.value != 0):
            file_path = '/root/keras-test/test-formatted/' + str(num.value) + '.jpg'
            img = image.load_img(file_path, target_size=(28, 28))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            y = automodel.predict(x)
            result.value = y.argmax(axis=-1)
            print(result.value)
            num.value = 0

num=multiprocessing.Value("i",0)
result=multiprocessing.Value("i",0)
needupdate=multiprocessing.Value("i",0)
recg_process = multiprocessing.Process(target = recognition_handler, args = (num,result,needupdate,))
recg_process.start()
kafka_process = multiprocessing.Process(target = kafka_handler, args = (needupdate,))
kafka_process.start()

result_str = ["unknown picture", "dinosaur", "elephant", "flower"]
@socketio.on('request',namespace='/testnamespace')
def give_response(data):
    msg_type = data.get('type')
    msg_data = data.get('data')

    if (msg_type == "ADDPERSON_REQ"):
        num.value = int(msg_data)
        while(True):
            if (num.value == 0):
                emit('response',{'code':'200','msg': "This is a " + result_str[result.value]})
                break

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=7000)
