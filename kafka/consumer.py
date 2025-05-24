from kafka import KafkaConsumer
import json
import time
from datetime import datetime
import os

BATCH_SIZE = 100
TIME_WINDOW = 30  # seconds
TOPIC_NAME = 'recipe_topic'

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers='kafka:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='recipe_group'
)

batch = []
start_time = time.time()

def save_batch(batch, index):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('../dataset/batch_output', exist_ok=True)
    filename = f"../dataset/batch_output/recipes_batch_{timestamp}_{index}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(batch)} records to {filename}")

batch_index = 1
for message in consumer:
    batch.append(message.value)

    if len(batch) >= BATCH_SIZE or (time.time() - start_time) >= TIME_WINDOW:
        save_batch(batch, batch_index)
        batch = []
        start_time = time.time()
        batch_index += 1
