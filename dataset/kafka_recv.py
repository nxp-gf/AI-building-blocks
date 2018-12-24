from kafka import KafkaConsumer
from kafka import KafkaProducer

#producer = KafkaProducer(bootstrap_servers='10.193.20.98:9092')

topic_training = "NXP_DATASET_TRAINING"
topic_training_quickly = "NXP_DATASET_QUICK_TRAINING"

#producer.send(topic_updated, b"modelname")


consumer = KafkaConsumer(bootstrap_servers='10.193.20.98:9092')
consumer.subscribe([topic_training_quickly])

for msg in consumer:
        print(msg)
        print(msg.value)
