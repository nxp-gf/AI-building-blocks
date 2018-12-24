# -*- coding: utf-8 -*-
import os
import tarfile
import base64

from flask import Flask, render_template, request, redirect,url_for, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, SubmitField
from wtforms.validators import Required
from flask_restful import Resource, Api
from kafka import KafkaConsumer
from kafka import KafkaProducer
import logging
import time

#UPLOAD_FOLDER = '/tftpboot/b33228/dataset/'
UPLOAD_FOLDER = './dataset/'
app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()
#app.config['UPLOADED_PHOTOS_DEST'] = UPLOAD_FOLDER
#app.config['UPLOADED_PHOTOS_URL'] = UPLOAD_FOLDER
#app.config['UPLOADED_DEFAULT_DEST'] = UPLOAD_FOLDER
#app.config['UPLOADED_DEFAULT_URL'] = UPLOAD_FOLDER
api = Api(app)

producer = KafkaProducer(bootstrap_servers='10.193.20.98:9092')
consumer = KafkaConsumer(bootstrap_servers='10.193.20.98:9092')
topic_training = "NXP_DATASET_TRAINING"
topic_training_quickly = "NXP_DATASET_QUICK_TRAINING"

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
    photo = FileField('File:')
    label = StringField('Label:')
    submit = SubmitField('submit')

class TrainingForm(FlaskForm):
    submit = SubmitField('training')

class QuickTrainingForm(FlaskForm):
    submit = SubmitField('quick training')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    train_form = TrainingForm()
    quick_form = QuickTrainingForm()
    if form.validate_on_submit():
        path = os.path.join(UPLOAD_FOLDER, form.label.data)
        print path
        if not os.path.exists(path):
            os.makedirs(path)
        print form.label.data
        filename = photos.save(form.photo.data, path)
        print filename
        file_url = photos.url(filename)
        print file_url
    else:
        file_url = None
    
    return render_template('index.html', form=form, train_form=train_form, 
            quick_form=quick_form, file_url=file_url)

@app.route('/train', methods=['GET', 'POST'])
def train():
    form = UploadForm()
    train_form = TrainingForm()
    quick_form = QuickTrainingForm()
    if request.method=='POST':
        print "send train"
        future = producer.send(topic_training, b"training")
        result = future.get(timeout=60)
        print "send ok"
    else:
        print "no train"
    
    return render_template('index.html', form=form, train_form=train_form, 
            quick_form=quick_form)

@app.route('/quick_train', methods=['GET', 'POST'])
def quick_train():
    form = UploadForm()
    train_form = TrainingForm()
    quick_form = QuickTrainingForm()
    if request.method=='POST':
        print "send quick train"
        future = producer.send(topic_training_quickly, b"quick training")
        result = future.get(timeout=60)
        print "send ok"

    return render_template('index.html', form=form, train_form=train_form, 
            quick_form=quick_form)

# RESTful API
class DataSendApi(Resource):
    def get(self):
        with tarfile.open("/tmp/tartest.tgz","w:gz") as tar:
            for root,dir,files in os.walk('/home/xulei/flask/upload/dataset/'):
                for file in files:
                    fullpath = os.path.join(root,file)
                    tar.add(fullpath)
        #with open("/tmp/tartest.tgz", "rb") as image_file:
        return send_from_directory('/tmp/', 'tartest.tgz', as_attachment=True)

api.add_resource(DataSendApi, '/dataset')

if __name__ == '__main__':
    app.run(host = '0.0.0.0')
