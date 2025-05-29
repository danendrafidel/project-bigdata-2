# model/train_model.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col, size # Pastikan 'size' diimpor
from pyspark.sql.types import ArrayType, StringType
import os
import glob
import json
import re # Impor modul 're' untuk regular expressions

# Konfigurasi
SPARK_APP_NAME = "RecipeClustering"
SPARK_MASTER = "local[*]"
BATCH_DATA_PATH = "dataset/batch_output/"
MODEL_OUTPUT_PATH = "models_output/"
K_VALUE_FOR_KMEANS = 15 # Jumlah cluster yang diinginkan

if not os.path.exists(MODEL_OUTPUT_PATH):
    os.makedirs(MODEL_OUTPUT_PATH)

def create_spark_session():
    return SparkSession.builder.appName(SPARK_APP_NAME).master(SPARK_MASTER).getOrCreate()

# UDF untuk membersihkan 'ingredients'
def clean_ingredients_list_string(ingredients_str):
    if not ingredients_str:
        return []
    try:
        list_of_ingredient_phrases = json.loads(ingredients_str)
        all_words = []
        for phrase in list_of_ingredient_phrases:
            if not isinstance(phrase, str):
                continue
            cleaned_phrase = phrase.lower()
            cleaned_phrase = re.sub(r"[^a-z\s]", "", cleaned_phrase) # Hapus non-alfabet kecuali spasi
            cleaned_phrase = re.sub(r"\s+", " ", cleaned_phrase).strip() # Hapus spasi berlebih
            words_in_phrase = cleaned_phrase.split()
            for word in words_in_phrase:
                if len(word) > 2: # Hanya ambil kata dengan panjang lebih dari 2
                    all_words.append(word)
        return all_words
    except Exception as e:
        # print(f"Error parsing ingredients: '{ingredients_str}', Error: {e}") # Aktifkan untuk debug
        return []

clean_ingredients_udf = udf(clean_ingredients_list_string, ArrayType(StringType()))

if __name__ == "__main__":
    spark = create_spark_session()

    # 1. Dapatkan daftar semua file batch dan urutkan
    json_files_glob_pattern = os.path.join(BATCH_DATA_PATH, "recipes_batch_*.json")
    all_batch_files = sorted(glob.glob(json_files_glob_pattern))

    if not all_batch_files:
        print(f"No batch files found at {json_files_glob_pattern}. Exiting.")
        spark.stop()
        exit()

    num_total_files = len(all_batch_files)
    print(f"Found {num_total_files} batch files: {all_batch_files}")

    # 2. Tentukan konfigurasi file untuk setiap model (Skema B: kumulatif)
    model_configurations = [] 

    if num_total_files > 0:
        # Model 1: Menggunakan sekitar 1/3 file pertama (minimal 1 file)
        end_idx_m1 = max(1, num_total_files // 3)
        model_configurations.append(all_batch_files[:end_idx_m1])

        # Model 2: Menggunakan sekitar 2/3 file pertama (kumulatif)
        # Hanya buat jika berbeda dari Model 1 dan ada cukup file
        end_idx_m2 = max(end_idx_m1, (2 * num_total_files) // 3) # Mulai dari end_idx_m1 agar tidak kurang
        if end_idx_m2 > end_idx_m1 and end_idx_m2 <= num_total_files:
            model_configurations.append(all_batch_files[:end_idx_m2])
        
        # Model 3: Menggunakan semua file (kumulatif)
        # Hanya buat jika berbeda dari konfigurasi model terakhir
        if not model_configurations or len(all_batch_files) > len(model_configurations[-1]):
             # Pastikan jika hanya 1 file, tidak duplikat jika logic di atas sudah mencakupnya
            if len(all_batch_files) > 0 and (len(model_configurations) == 0 or all_batch_files != model_configurations[-1]):
                 model_configurations.append(all_batch_files)


    # Hapus konfigurasi duplikat (jika num_total_files sangat kecil, bisa terjadi duplikasi)
    unique_model_configs_paths = []
    seen_config_tuples = set()
    for config_list in model_configurations:
        # Untuk perbandingan, ubah list path file menjadi tuple agar bisa dimasukkan ke set
        config_tuple_representation = tuple(sorted(config_list))
        if config_tuple_representation not in seen_config_tuples:
            unique_model_configs_paths.append(config_list)
            seen_config_tuples.add(config_tuple_representation)
    
    print(f"Will attempt to train {len(unique_model_configs_paths)} model(s) based on available data.")

    for i, files_for_this_model in enumerate(unique_model_configs_paths):
        model_version_num = i + 1

        if not files_for_this_model:
            print(f"Skipping model v{model_version_num}, no files selected (this shouldn't happen with current logic).")
            continue

        print(f"\n--- Training Model v{model_version_num} using {len(files_for_this_model)} batch files ---")
        print(f"Files: {files_for_this_model}")

        # Baca data untuk model saat ini
        raw_df_model = spark.read.option("multiLine", "true").json(files_for_this_model)
        
        # print(f"Schema for Model v{model_version_num}:")
        # raw_df_model.printSchema()

        if 'ingredients' not in raw_df_model.columns:
            print(f"ERROR: Column 'ingredients' not found for Model v{model_version_num}.")
            continue

        data_df_model = raw_df_model.filter(col("ingredients").isNotNull())
        data_df_model = data_df_model.withColumn("cleaned_ingredients", clean_ingredients_udf(col("ingredients")))
        
        processed_df_model = data_df_model.filter(size(col("cleaned_ingredients")) > 0)
        
        num_data_points = processed_df_model.count()
        print(f"Model v{model_version_num}: Rows after UDF and filtering empty ingredients: {num_data_points}")

        if num_data_points < K_VALUE_FOR_KMEANS:
            print(f"WARNING: Model v{model_version_num} - Not enough data points ({num_data_points}) to train K-Means with k={K_VALUE_FOR_KMEANS}. Skipping this model.")
            continue

        # Definisikan pipeline
        
        remover = StopWordsRemover(inputCol="cleaned_ingredients", outputCol="filtered_ingredients")
        hashingTF = HashingTF(inputCol="filtered_ingredients", outputCol="raw_features", numFeatures=2000)
        idf = IDF(inputCol="raw_features", outputCol="features")
        kmeans = KMeans(featuresCol="features", k=K_VALUE_FOR_KMEANS, seed=1)
        pipeline = Pipeline(stages=[remover, hashingTF, idf, kmeans])
        
        print(f"Starting training for Model v{model_version_num}...")
        try:
            model = pipeline.fit(processed_df_model)
            print(f"Training finished for Model v{model_version_num}.")
            
            model_name = f"recipe_cluster_model_v{model_version_num}"
            model_path = os.path.join(MODEL_OUTPUT_PATH, model_name)
            # Hapus direktori model jika sudah ada (untuk menimpa)
            # Anda mungkin ingin strategi yang lebih baik di produksi
            import shutil
            if os.path.exists(model_path):
                print(f"Removing existing model directory: {model_path}")
                shutil.rmtree(model_path)

            model.save(model_path)
            print(f"Model {model_name} saved to {model_path}")

            # (Opsional) Tampilkan prediksi untuk model ini
            # predictions = model.transform(processed_df_model)
            # predictions.select("title", "cleaned_ingredients", "prediction").show(3, truncate=False)
        except Exception as e_train:
            print(f"ERROR during training or saving Model v{model_version_num}: {e_train}")


    print("\nAll model training attempts finished.")
    spark.stop()