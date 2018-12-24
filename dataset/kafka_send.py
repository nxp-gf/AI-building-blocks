from kafka import KafkaConsumer
from kafka import KafkaProducer
import logging

#logging.basicConfig(level=logging.DEBUG)
producer = KafkaProducer(bootstrap_servers='10.193.20.98:9092')

topic_training = "NXP_DATASET_TRAINING"
topic_training_quickly = "NXP_DATASET_QUICK_TRAINING"

producer.send(topic_training, b"in training")
import time
time.sleep(1)


#consumer = KafkaConsumer(bootstrap_servers='10.193.20.98:9092')
#consumer.subscribe([topic_training])

#for msg in consumer:
#        print(msg)
