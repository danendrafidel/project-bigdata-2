from kafka import KafkaProducer
import csv
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC_NAME = 'recipe_topic'
CSV_FILE = '../dataset/recipes_data.csv'

with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        producer.send(TOPIC_NAME, value=row)
        print("Sent:", row)
        time.sleep(random.uniform(0.1, 1.0))  # Simulasi streaming

producer.flush()
