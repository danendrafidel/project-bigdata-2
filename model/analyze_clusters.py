# analyze_clusters.py (atau tambahkan ke train_model.py setelah training model_v2)

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf, col, size
from pyspark.sql.types import ArrayType, StringType
import os
import json
import re
import glob

# --- Konfigurasi (Sama seperti train_model.py) ---
SPARK_APP_NAME = "ClusterAnalysis"
SPARK_MASTER = "local[*]"
BATCH_DATA_PATH = "../dataset/batch_output/" # Path ke data batch asli
MODEL_TO_ANALYZE_PATH = "../models_output/recipe_cluster_model_v3" # Path ke model v2

# --- UDF (HARUS SAMA PERSIS dengan yang digunakan saat training model_v2) ---
def clean_ingredients_list_string(ingredients_str):
    # ... (salin UDF yang sama persis dari train_model.py) ...
    if not ingredients_str: return []
    try:
        list_of_ingredient_phrases = json.loads(ingredients_str)
        all_words = []
        for phrase in list_of_ingredient_phrases:
            if not isinstance(phrase, str): continue
            cleaned_phrase = phrase.lower()
            cleaned_phrase = re.sub(r"[^a-z\s]", "", cleaned_phrase)
            cleaned_phrase = re.sub(r"\s+", " ", cleaned_phrase).strip()
            words_in_phrase = cleaned_phrase.split()
            for word in words_in_phrase:
                if len(word) > 2: all_words.append(word)
        return all_words
    except: return []

clean_ingredients_udf = udf(clean_ingredients_list_string, ArrayType(StringType()))

def get_spark_session():
    return SparkSession.builder.appName(SPARK_APP_NAME).master(SPARK_MASTER).getOrCreate()

if __name__ == "__main__":
    spark = get_spark_session()

    # 1. Muat model_v2 yang sudah dilatih
    print(f"Loading model from: {MODEL_TO_ANALYZE_PATH}")
    try:
        loaded_model_v2 = PipelineModel.load(MODEL_TO_ANALYZE_PATH)
        print("Model v2 loaded successfully.")
    except Exception as e:
        print(f"Error loading model v2: {e}")
        spark.stop()
        exit()

    # 2. Baca data yang DIGUNAKAN untuk melatih model_v2
    #    Ini penting! Anda perlu data training asli dari model_v2.
    #    Asumsikan model_v2 dilatih dengan 3 file batch pertama (sesuai run Anda sebelumnya)
    #    Anda mungkin perlu menyesuaikan path file ini jika berbeda.
    files_for_model_v2 = sorted(glob.glob(os.path.join(BATCH_DATA_PATH, "recipes_batch_*.json")))[:3] # Ambil 3 file pertama
    
    if not files_for_model_v2:
        print("No batch files found for model_v2 training data.")
        spark.stop()
        exit()
        
    print(f"Reading training data for model_v2 from files: {files_for_model_v2}")
    raw_training_data_v2 = spark.read.option("multiLine", "true").json(files_for_model_v2)

    # 3. Lakukan preprocessing pada data training ini (UDF)
    #    Tidak perlu menjalankan ulang HashingTF, IDF karena itu bagian dari pipeline model.
    #    Kita hanya perlu kolom 'cleaned_ingredients' untuk model.transform()
    if 'ingredients' not in raw_training_data_v2.columns:
        print("Column 'ingredients' not found in training data.")
        spark.stop()
        exit()
        
    processed_training_data_v2 = raw_training_data_v2.filter(col("ingredients").isNotNull())
    processed_training_data_v2 = processed_training_data_v2.withColumn("cleaned_ingredients", clean_ingredients_udf(col("ingredients")))
    processed_training_data_v2 = processed_training_data_v2.filter(size(col("cleaned_ingredients")) > 0)

    print(f"Number of data points in model_v2 training set: {processed_training_data_v2.count()}")

    # 4. Dapatkan prediksi cluster untuk setiap data point di data training model_v2
    predictions_on_training_data = loaded_model_v2.transform(processed_training_data_v2)
    
    print("\nPredictions on model_v2 training data (sample):")
    predictions_on_training_data.select("title", "cleaned_ingredients", "prediction").show(10, truncate=False)

    # 5. Filter untuk melihat resep-resep yang masuk ke Cluster 0
    CLUSTER_TO_ANALYZE = 0
    recipes_in_cluster_0 = predictions_on_training_data.filter(col("prediction") == CLUSTER_TO_ANALYZE)

    print(f"\n--- Recipes in Cluster {CLUSTER_TO_ANALYZE} (from model_v2 training data) ---")
    if recipes_in_cluster_0.count() > 0:
        recipes_in_cluster_0.select("title", "ingredients").show(truncate=False) # Tampilkan semua jika sedikit, atau .show(20, truncate=False)
        print(f"Total recipes in Cluster {CLUSTER_TO_ANALYZE}: {recipes_in_cluster_0.count()}")
        
        # Opsional: Analisis bahan paling umum di Cluster 0
        # from pyspark.sql.functions import explode, count
        # common_ingredients_in_cluster_0 = recipes_in_cluster_0.select(explode(col("cleaned_ingredients")).alias("ingredient")) \
        #                                                     .groupBy("ingredient") \
        #                                                     .agg(count("*").alias("frequency")) \
        #                                                     .orderBy(col("frequency").desc())
        # print("\nMost common cleaned ingredients in Cluster 0:")
        # common_ingredients_in_cluster_0.show()
    else:
        print(f"No recipes found in Cluster {CLUSTER_TO_ANALYZE}.")

    spark.stop()