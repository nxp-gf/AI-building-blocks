import os
import multiprocessing

TEST_IMG_DIR_INPUT = "/root/re/test"
TEST_IMG_DIR_OUTPUT = "/root/re/test-formatted"
TRAIN_IMG_DIR_INPUT = "/root/re/train"
TRAIN_IMG_DIR_OUTPUT = "/root/re/train-formatted"
IMAGE_SIZE = 28
TRAIN_CSV_DIR = '/root/re/train_labels.csv'
TRAIN_IMG_DIR = '/root/re/train-formatted'
TEST_CSV_DIR = '/root/re/test_labels.csv'
TEST_IMG_DIR = '/root/re/test-formatted'
MODEL_DIR = '/root/re/my_model.h4'
trn_path = '/root/re/train-ge'
val_path = '/root/re/test-ge'

def train(filename):
    from tensorflow.keras.preprocessing import image
    from autokeras.image.image_supervised import load_image_dataset, ImageClassifier
    from keras.models import load_model
    from keras.utils import plot_model
    from keras.preprocessing.image import load_img, img_to_array
    from keras.models import load_model
    from keras.layers import Activation
    from keras.preprocessing.image import ImageDataGenerator
    import numpy as np

    def format_img(input_dir, output_dir):
        for file_name in os.listdir(input_dir):
            path_name = os.path.join(input_dir, file_name)
            img = image.load_img(path_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            path_name = os.path.join(output_dir, file_name)
            img.save(path_name)

    print("Training started")
    format_img(TEST_IMG_DIR_INPUT, TEST_IMG_DIR_OUTPUT)
    format_img(TRAIN_IMG_DIR_INPUT, TRAIN_IMG_DIR_OUTPUT)
    train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=TRAIN_IMG_DIR)
    test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=TEST_IMG_DIR)
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    print("train data shape:", train_data.shape)
    clf = ImageClassifier(verbose=True)
    clf.fit(train_data, train_labels, time_limit=2 * 60)
    clf.final_fit(train_data, train_labels, test_data, test_labels, retrain=True)
    clf.export_keras_model(MODEL_DIR)
    y = clf.evaluate(test_data, test_labels)
    print("evaluate:", y)
    automodel = load_model(MODEL_DIR)
    x = automodel.output
    x = Activation('softmax', name='activation_add')(x)
    from keras.layers import Model
    automodel = Model(automodel.input, x)
    automodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    generator = ImageDataGenerator()
    trn_data = generator.flow_from_directory(trn_path, batch_size=32, target_size=(28, 28))
    val_data = generator.flow_from_directory(val_path, batch_size=32, target_size=(28, 28))
    automodel.fit_generator(trn_data,200,10)
    automodel.save(filename)
    print("Training ended")

def kafka_handler():
    from kafka import KafkaConsumer
    from kafka import KafkaProducer

    topic_updated = "NXP_MODELS_UPDATED"
    topic_training = "NXP_DATASET_TRAINING"
    topic_training_quickly = "NXP_DATASET_QUICK_TRAINING"
    consumer = KafkaConsumer(bootstrap_servers='10.193.20.98:9092')
    producer = KafkaProducer(bootstrap_servers='10.193.20.98:9092')
    consumer.subscribe([topic_training, topic_training_quickly])
    for msg in consumer:
        if (msg.topic == topic_training):
            filename = "/root/building-blocks/finalmodel"
            train(filename)
            future = producer.send(topic_updated, filename.encode())
            result = future.get(timeout=60)
        elif (msg.topic == topic_training_quickly):
            filename = "/root/building-blocks/testmodel"
            future = producer.send(topic_updated, filename.encode())
            result = future.get(timeout=60)
        else:
            print("Invilid msg topic:", msg.topic)

train_process = multiprocessing.Process(target = kafka_handler, args = ())
train_process.start()

from flask import Flask, send_from_directory
from flask_restful import reqparse, abort, Api, Resource, request

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
