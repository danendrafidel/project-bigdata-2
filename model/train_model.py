# model/train_model.py

import os
import sys
import glob
import json
import re # Impor modul 're' untuk regular expressions
import shutil # Untuk menghapus direktori model yang ada

# ---------------------------------------------------------------------------
# PENTING: Atur environment variable PYSPARK_PYTHON dan PYSPARK_DRIVER_PYTHON
# sebelum mengimpor modul pyspark apa pun.
# Ini memastikan Spark menggunakan interpreter Python dari virtual environment Anda.
# ---------------------------------------------------------------------------
if 'PYSPARK_PYTHON' not in os.environ:
    os.environ['PYSPARK_PYTHON'] = sys.executable
if 'PYSPARK_DRIVER_PYTHON' not in os.environ:
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Sekarang baru impor modul PySpark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col, size # Pastikan 'size' diimpor
from pyspark.sql.types import ArrayType, StringType

# Konfigurasi
SPARK_APP_NAME = "RecipeClustering"
SPARK_MASTER = "local[*]" # Gunakan semua core yang tersedia
BATCH_DATA_PATH = "dataset/batch_output/"
MODEL_OUTPUT_PATH = "models_output/"
K_VALUE_FOR_KMEANS = 10 # Jumlah cluster yang diinginkan

# Pastikan direktori output model ada
if not os.path.exists(MODEL_OUTPUT_PATH):
    os.makedirs(MODEL_OUTPUT_PATH)
    print(f"Created directory: {MODEL_OUTPUT_PATH}")

def create_spark_session():
    """Membuat dan mengembalikan SparkSession."""
    print(f"Attempting to create SparkSession with PYSPARK_PYTHON='{os.environ.get('PYSPARK_PYTHON')}'")
    return SparkSession.builder.appName(SPARK_APP_NAME).master(SPARK_MASTER).getOrCreate()

# UDF untuk membersihkan 'ingredients'
def clean_ingredients_list_string(ingredients_str):
    """Membersihkan string JSON dari ingredients menjadi daftar kata-kata bersih."""
    if not ingredients_str:
        return []
    try:
        # Coba parsing sebagai JSON, jika gagal, mungkin sudah berupa list string (jarang)
        # atau format tak terduga.
        if isinstance(ingredients_str, str):
            list_of_ingredient_phrases = json.loads(ingredients_str)
        elif isinstance(ingredients_str, list): # Jika input sudah berupa list
            list_of_ingredient_phrases = ingredients_str
        else:
            return [] # Format tidak dikenali

        all_words = []
        for phrase in list_of_ingredient_phrases:
            if not isinstance(phrase, str): # Lewati jika frasa bukan string
                continue
            cleaned_phrase = phrase.lower()
            cleaned_phrase = re.sub(r"[^a-z\s]", "", cleaned_phrase) # Hapus non-alfabet kecuali spasi
            cleaned_phrase = re.sub(r"\s+", " ", cleaned_phrase).strip() # Hapus spasi berlebih
            words_in_phrase = cleaned_phrase.split()
            for word in words_in_phrase:
                if len(word) > 2: # Hanya ambil kata dengan panjang lebih dari 2
                    all_words.append(word)
        return all_words
    except json.JSONDecodeError:
        # Jika ini adalah string tunggal yang bukan JSON, coba proses langsung
        if isinstance(ingredients_str, str):
            cleaned_phrase = ingredients_str.lower()
            cleaned_phrase = re.sub(r"[^a-z\s]", "", cleaned_phrase)
            cleaned_phrase = re.sub(r"\s+", " ", cleaned_phrase).strip()
            words_in_phrase = cleaned_phrase.split()
            all_words = [word for word in words_in_phrase if len(word) > 2]
            return all_words
        # print(f"Error parsing ingredients: '{ingredients_str}', Error: {e}") # Aktifkan untuk debug
        return []
    except Exception: # Tangkap kesalahan lain yang mungkin terjadi
        # print(f"Unexpected error processing ingredients: '{ingredients_str}', Error: {e}")
        return []

# Daftarkan UDF
clean_ingredients_udf = udf(clean_ingredients_list_string, ArrayType(StringType()))

if __name__ == "__main__":
    spark = None # Inisialisasi spark ke None
    try:
        spark = create_spark_session()
        print("SparkSession created successfully.")

        # 1. Dapatkan daftar semua file batch dan urutkan
        json_files_glob_pattern = os.path.join(BATCH_DATA_PATH, "recipes_batch_*.json")
        all_batch_files = sorted(glob.glob(json_files_glob_pattern))

        if not all_batch_files:
            print(f"No batch files found at {json_files_glob_pattern}. Exiting.")
            if spark:
                spark.stop()
            exit()

        num_total_files = len(all_batch_files)
        print(f"Found {num_total_files} batch files.") # Disingkat agar tidak terlalu verbose jika banyak file
        # print(f"Found {num_total_files} batch files: {all_batch_files}") # Uncomment jika ingin lihat list file

        # 2. Tentukan konfigurasi file untuk setiap model (Skema B: kumulatif)
        model_configurations = []

        if num_total_files > 0:
            # Model 1: Menggunakan sekitar 1/3 file pertama (minimal 1 file)
            end_idx_m1 = max(1, num_total_files // 3)
            model_configurations.append(all_batch_files[:end_idx_m1])

            # Model 2: Menggunakan sekitar 2/3 file pertama (kumulatif)
            end_idx_m2 = max(end_idx_m1, (2 * num_total_files) // 3)
            if end_idx_m2 > end_idx_m1 and end_idx_m2 <= num_total_files:
                model_configurations.append(all_batch_files[:end_idx_m2])
            
            # Model 3: Menggunakan semua file (kumulatif)
            if not model_configurations or len(all_batch_files) > len(model_configurations[-1]):
                if len(all_batch_files) > 0 and (not model_configurations or all_batch_files != model_configurations[-1]):
                    model_configurations.append(all_batch_files)

        # Hapus konfigurasi duplikat
        unique_model_configs_paths = []
        seen_config_tuples = set()
        for config_list in model_configurations:
            config_tuple_representation = tuple(sorted(config_list))
            if config_tuple_representation not in seen_config_tuples:
                unique_model_configs_paths.append(config_list)
                seen_config_tuples.add(config_tuple_representation)
        
        print(f"Will attempt to train {len(unique_model_configs_paths)} model(s) based on available data.")

        for i, files_for_this_model in enumerate(unique_model_configs_paths):
            model_version_num = i + 1

            if not files_for_this_model:
                print(f"Skipping model v{model_version_num}, no files selected.")
                continue

            print(f"\n--- Training Model v{model_version_num} using {len(files_for_this_model)} batch files ---")
            # print(f"Files: {files_for_this_model}") # Uncomment jika ingin lihat list file untuk model ini

            # Baca data untuk model saat ini
            raw_df_model = spark.read.option("multiLine", "true").json(files_for_this_model)
            
            if 'ingredients' not in raw_df_model.columns:
                print(f"ERROR: Column 'ingredients' not found for Model v{model_version_num}. Skipping.")
                continue

            # Filter baris di mana 'ingredients' adalah null sebelum UDF
            data_df_model = raw_df_model.filter(col("ingredients").isNotNull())
            
            # Terapkan UDF dan filter lagi jika UDF menghasilkan array kosong
            data_df_model = data_df_model.withColumn("cleaned_ingredients", clean_ingredients_udf(col("ingredients")))
            processed_df_model = data_df_model.filter(size(col("cleaned_ingredients")) > 0)
            
            num_data_points = processed_df_model.count() # Ini adalah aksi Spark pertama yang memicu eksekusi
            print(f"Model v{model_version_num}: Rows after UDF and filtering empty ingredients: {num_data_points}")

            if num_data_points == 0:
                print(f"WARNING: Model v{model_version_num} - No data points after processing. Skipping this model.")
                continue
            if num_data_points < K_VALUE_FOR_KMEANS:
                print(f"WARNING: Model v{model_version_num} - Not enough data points ({num_data_points}) "
                      f"to train K-Means with k={K_VALUE_FOR_KMEANS}. Skipping this model.")
                continue

            # Definisikan pipeline ML
            remover = StopWordsRemover(inputCol="cleaned_ingredients", outputCol="filtered_ingredients")
            hashingTF = HashingTF(inputCol="filtered_ingredients", outputCol="raw_features", numFeatures=2000)
            idf = IDF(inputCol="raw_features", outputCol="features")
            kmeans = KMeans(featuresCol="features", k=K_VALUE_FOR_KMEANS, seed=1) # seed untuk reproduktifitas
            pipeline = Pipeline(stages=[remover, hashingTF, idf, kmeans])
            
            print(f"Starting training for Model v{model_version_num}...")
            model = pipeline.fit(processed_df_model) # Aksi Spark kedua
            print(f"Training finished for Model v{model_version_num}.")
            
            model_name = f"recipe_cluster_model_v{model_version_num}"
            model_path = os.path.join(MODEL_OUTPUT_PATH, model_name)
            
            # Hapus direktori model jika sudah ada untuk menimpa
            if os.path.exists(model_path):
                print(f"Removing existing model directory: {model_path}")
                shutil.rmtree(model_path)

            model.save(model_path) # Aksi Spark ketiga (jika melibatkan penulisan data, bisa jadi)
            print(f"Model {model_name} saved to {model_path}")

            # (Opsional) Tampilkan prediksi untuk sampel data model ini
            # predictions = model.transform(processed_df_model)
            # print(f"Sample predictions for Model v{model_version_num}:")
            # predictions.select("title", "cleaned_ingredients", "prediction").show(5, truncate=False)

    except Exception as e_main:
        print(f"An critical error occurred in the main script: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping SparkSession...")
            spark.stop()
            print("SparkSession stopped.")
        print("All model training attempts finished or script terminated.")