# project-bigdata-2

## Langkah-Langkah Menjalankan Proyek

### 1. Setup Virtual Environment Python

Diasumsikan Anda sudah mengatur `pyenv local 3.8.18` di root direktori proyek.

```bash
# (Dari root direktori proyek)
# Hapus venv lama jika ada (opsional, untuk fresh start)
# rm -rf venv

# Buat virtual environment baru dengan Python 3.8
python -m venv venv

# Aktifkan virtual environment
source venv/bin/activate  # untuk Linux/Mac
.\venv\Scripts\activate  # untuk Windows

# Instal dependensi
pip install --upgrade pip
pip install kafka-python pandas pyspark
```

### 2. Jalankan Kafka & Zookeeper via Docker Compose

(Dari root direktori proyek, di terminal terpisah)

```bash
docker-compose up -d
```

### 3. Buat Kafka Topic

(Dari root direktori proyek, di terminal yang sama dengan docker-compose atau terminal baru)

```bash
docker-compose exec kafka kafka-topics.sh --create \
  --topic recipe_topic \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1
```

### 4. Jalankan Kafka Consumer

Consumer akan mendengarkan pesan dan menyimpan batch data.
(Buka Terminal BARU, dari root direktori proyek)

```bash
source venv/bin/activate
python kafka/consumer.py
```

### 5. Jalankan Kafka Producer

Producer akan membaca dataset dan mengirimkannya ke Kafka.
(Buka Terminal BARU LAINNYA, dari root direktori proyek)

```bash
source venv/bin/activate
python kafka/producer.py
```

### 6. Jalankan Script Spark untuk Melatih Model

Script ini akan membaca data batch yang disimpan oleh consumer, melakukan preprocessing, dan melatih model K-Means.
(Dari root direktori proyek, di terminal baru atau yang sudah ada)

```bash
source venv/bin/activate

# Pastikan direktori output model ada
mkdir -p models_output

# Jalankan script training model
python model/train_model.py
```
