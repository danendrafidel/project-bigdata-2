# project-bigdata-2

## Anggota Kelompok

| Nama                  | NRP        |
| --------------------- | ---------- |
| Maulana Ahmad Zahiri  | 5027231010 |
| Danendra Fidel Khansa | 5027231063 |
| Dimas Andhika DIputra | 5027231074 |

## Langkah-Langkah Menjalankan Proyek

### Setup Apache Kafka dan Zookeeper

Instalasi dapat dilakukan dengan menggunakan file `docker-compose.yml` dengan konfigurasi berikut:

```yml
services:
  zookeeper:
    image: "bitnami/zookeeper:3.8" # Versi spesifik untuk zookeeper juga baik
    container_name: zookeeper
    environment:
      - ALLOW_ANONYMOUS_LOGIN=yes
    ports:
      - "2181:2181"

  kafka:
    image: "bitnami/kafka:3.3.2" # Coba versi ini dulu
    container_name: kafka
    ports:
      - "9092:9092"
    environment:
      # Konfigurasi untuk mode Zookeeper
      - KAFKA_BROKER_ID=1
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 # Penting untuk mode Zookeeper
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
      - KAFKA_LISTENERS=PLAINTEXT://:9092 # Untuk internal container
      - ALLOW_PLAINTEXT_LISTENER=yes # Biasanya untuk kompatibilitas
    depends_on:
      - zookeeper
```

### 1. Setup Virtual Environment Python

Diasumsikan Anda sudah mengatur `pyenv local 3.8.18` di root direktori proyek.

```bash
# (Dari root direktori proyek)
Hapus venv lama jika ada (opsional, untuk fresh start)
rm -rf venv

# Buat virtual environment baru dengan Python 3.8
python -m venv venv

# Aktifkan virtual environment
source venv/bin/activate  # untuk Linux/Mac
source venv/Scripts/activate # untuk Windows

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
# Masuk ke image dan bikin topic
docker-compose exec kafka kafka-topics.sh --create \
  --topic recipe_topic \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1

# Cek apakah topic tersedia
kafka-topics.sh --list --bootstrap-server localhost:9092
```

Dokumentasi:

![Topic](<images/Topic (1).png>)

![Topic](<images/Topic (2).png>)

### 4. Jalankan Kafka Consumer

Consumer akan mendengarkan pesan dan menyimpan batch data.
(Buka Terminal BARU, dari root direktori proyek)

```bash
source venv/bin/activate  # untuk Linux/Mac
source venv/Scripts/activate # untuk Windows
python kafka/consumer.py
```

### 5. Jalankan Kafka Producer

Producer akan membaca dataset dan mengirimkannya ke Kafka.
(Buka Terminal BARU LAINNYA, dari root direktori proyek)

```bash
source venv/bin/activate  # untuk Linux/Mac
source venv/Scripts/activate # untuk Windows
python kafka/producer.py
```

Dokumentasi:

![Kafka](<images/Kafka (1).png>)

![Kafka](<images/Kafka (2).png>)

### 6. Jalankan Script Spark untuk Melatih Model

Script ini akan membaca data batch yang disimpan oleh consumer, melakukan preprocessing, dan melatih model K-Means.
(Dari root direktori proyek, di terminal baru atau yang sudah ada)

```bash
source venv/bin/activate  # untuk Linux/Mac
source venv/Scripts/activate # untuk Windows

# Install Pyspark yang kompatibel
pip install pyspark==3.5.1

# Pastikan direktori output model ada
mkdir -p models_output

# Jalankan script training model
python model/train_model.py
```

Dari folder models_output yang telah dibuat, akan muncul folder train `recipe_cluster_model`

Dokumentasi:

![Pyspark](<images/Pyspark (1).png)

![Pyspark](<images/Pyspark (2).png)

### 7. Jalankan Script API untuk Mengetes Cluster dari Dataset ini

```bash
# Jalankan script api
python api/app.py
```

### 8. Buka POSTMAN untuk API TESTnya

Berikut Dokumentasinya:

- Cek Informasi API (GET)
  ![Dokumentasi](images/Dokumentasi.png)

- Cek Informasi Model (GET)
  ![Dokumentasi](<images/Dokumentasi (2).png>)

- Input yang Salah (POST)

```json
{
  "bahan": "any, ingredients"
}
```

![Dokumentasi](<images/Dokumentasi (3).png>)

- Cluster 0

```json
{
  "ingredients": "romaine lettuce, cherry tomatoes, cucumber, red onion, feta cheese, Kalamata olives, extra virgin olive oil, red wine vinegar, Dijon mustard, dried oregano"
}
```

![Dokumentasi API](<images/Dokumentasi%20API%20(1).png>)

- Cluster 1

```json
{
  "ingredients": "all-purpose flour, unsweetened cocoa powder, baking soda, baking powder, salt, granulated sugar, eggs, milk, vegetable oil, vanilla extract, boiling water"
}
```

![Dokumentasi API](<images/Dokumentasi%20API%20(2).png>)

- Cluster 2

```json
{
  "ingredients": "spaghetti, ground beef, canned crushed tomatoes, onion, garlic, olive oil, basil, oregano, parmesan cheese"
}
```

![Dokumentasi API](<images/Dokumentasi%20API%20(3).png>)

- Cluster 3

```json
{
  "ingredients": "chicken pieces, all-purpose flour, paprika, garlic powder, onion powder, salt, black pepper, buttermilk, vegetable oil for frying"
}
```

![Dokumentasi API](<images/Dokumentasi%20API%20(4).png>)
