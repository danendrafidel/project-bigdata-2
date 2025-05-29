# model/train_model.py

import os
import sys
import glob
import json
import re
import shutil

# ... (bagian pengaturan PYSPARK_PYTHON tetap sama) ...
if 'PYSPARK_PYTHON' not in os.environ:
    os.environ['PYSPARK_PYTHON'] = sys.executable
if 'PYSPARK_DRIVER_PYTHON' not in os.environ:
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Word2Vec # Alternatif untuk HashingTF+IDF
from pyspark.ml.clustering import KMeans, BisectingKMeans # Alternatif untuk KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col, size
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.evaluation import ClusteringEvaluator # Untuk mengevaluasi model

# Konfigurasi
SPARK_APP_NAME = "RecipeClustering"
SPARK_MASTER = "local[*]"
BATCH_DATA_PATH = "dataset/batch_output/"
MODEL_OUTPUT_PATH = "models_output/"
K_VALUE_FOR_KMEANS = 10 # Anda bisa eksperimen dengan nilai K ini
HASHINGTF_NUM_FEATURES = 5000 # Naikkan sedikit, bisa dieksperimenkan
MIN_DATA_POINTS_FACTOR = 2 # Model akan dilatih jika num_data_points >= K_VALUE_FOR_KMEANS * MIN_DATA_POINTS_FACTOR

# ... (fungsi create_spark_session dan clean_ingredients_list_string tetap sama) ...
# Pastikan direktori output model ada
if not os.path.exists(MODEL_OUTPUT_PATH):
    os.makedirs(MODEL_OUTPUT_PATH)
    print(f"Created directory: {MODEL_OUTPUT_PATH}")

def create_spark_session():
    """Membuat dan mengembalikan SparkSession."""
    print(f"Attempting to create SparkSession with PYSPARK_PYTHON='{os.environ.get('PYSPARK_PYTHON')}'")
    # Tambahkan beberapa konfigurasi untuk stabilitas dan logging
    return SparkSession.builder.appName(SPARK_APP_NAME)\
        .master(SPARK_MASTER)\
        .config("spark.sql.shuffle.partitions", "4")\
        .config("spark.driver.memory", "2g")\
        .config("spark.executor.memory", "2g")\
        .getOrCreate() # Pastikan .getOrCreate() berada pada level indentasi yang sama atau sebagai kelanjutan langsung

def clean_ingredients_list_string(ingredients_str):
    """Membersihkan string JSON dari ingredients menjadi daftar kata-kata bersih."""
    if not ingredients_str:
        return []
    try:
        if isinstance(ingredients_str, str):
            list_of_ingredient_phrases = json.loads(ingredients_str)
        elif isinstance(ingredients_str, list):
            list_of_ingredient_phrases = ingredients_str
        else:
            return []

        all_words = []
        for phrase in list_of_ingredient_phrases:
            if not isinstance(phrase, str):
                continue
            # Lebih agresif dalam membersihkan, fokus pada kata bahan
            cleaned_phrase = phrase.lower()
            # Hapus ukuran, unit, dan teks dalam kurung (seringkali instruksi)
            cleaned_phrase = re.sub(r'\([^)]*\)', '', cleaned_phrase) # hapus (text)
            cleaned_phrase = re.sub(r'\b(oz|ounce|ounces|g|gram|grams|kg|kilogram|kilograms|lb|lbs|pound|pounds|cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|ml|milliliter|milliliters|l|liter|liters|to taste|or more|as needed|chopped|sliced|diced|minced|peeled|cored|seeded|fresh|dried|ground|optional|preferably|about|approximately)\b', '', cleaned_phrase, flags=re.IGNORECASE)
            cleaned_phrase = re.sub(r"[^a-z\s-]", "", cleaned_phrase) # Izinkan hyphen (misal: "self-raising")
            cleaned_phrase = re.sub(r"\s+", " ", cleaned_phrase).strip()
            
            words_in_phrase = cleaned_phrase.split()
            for word in words_in_phrase:
                # Hapus kata-kata yang sangat pendek atau umum yang mungkin lolos dari stopwords
                if len(word) > 2 and word not in ['and', 'the', 'for', 'with', 'into', 'onto', 'from']: 
                    all_words.append(word)
        return list(set(all_words)) # Kembalikan kata unik untuk menghindari bobot berlebih dari pengulangan di satu resep
    except json.JSONDecodeError:
        if isinstance(ingredients_str, str):
            cleaned_phrase = ingredients_str.lower()
            cleaned_phrase = re.sub(r'\([^)]*\)', '', cleaned_phrase)
            cleaned_phrase = re.sub(r'\b(oz|ounce|ounces|g|gram|grams|kg|kilogram|kilograms|lb|lbs|pound|pounds|cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|ml|milliliter|milliliters|l|liter|liters|to taste|or more|as needed|chopped|sliced|diced|minced|peeled|cored|seeded|fresh|dried|ground|optional|preferably|about|approximately)\b', '', cleaned_phrase, flags=re.IGNORECASE)
            cleaned_phrase = re.sub(r"[^a-z\s-]", "", cleaned_phrase)
            cleaned_phrase = re.sub(r"\s+", " ", cleaned_phrase).strip()
            words_in_phrase = cleaned_phrase.split()
            all_words = [word for word in words_in_phrase if len(word) > 2 and word not in ['and', 'the', 'for', 'with', 'into', 'onto', 'from']]
            return list(set(all_words))
        return []
    except Exception:
        return []

clean_ingredients_udf = udf(clean_ingredients_list_string, ArrayType(StringType()))

if __name__ == "__main__":
    spark = None
    try:
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("WARN") # Set log level setelah session dibuat
        print("SparkSession created successfully.")

        # ... (bagian glob file tetap sama) ...
        json_files_glob_pattern = os.path.join(BATCH_DATA_PATH, "recipes_batch_*.json")
        all_batch_files = sorted(glob.glob(json_files_glob_pattern))

        if not all_batch_files:
            print(f"No batch files found at {json_files_glob_pattern}. Exiting.")
            if spark:
                spark.stop()
            exit()

        num_total_files = len(all_batch_files)
        print(f"Found {num_total_files} batch files.")

        # ... (bagian konfigurasi model tetap sama) ...
        model_configurations = []
        if num_total_files > 0:
            end_idx_m1 = max(1, num_total_files // 3)
            model_configurations.append(all_batch_files[:end_idx_m1])
            end_idx_m2 = max(end_idx_m1, (2 * num_total_files) // 3)
            if end_idx_m2 > end_idx_m1 and end_idx_m2 <= num_total_files:
                model_configurations.append(all_batch_files[:end_idx_m2])
            if not model_configurations or len(all_batch_files) > len(model_configurations[-1]):
                if len(all_batch_files) > 0 and (not model_configurations or all_batch_files != model_configurations[-1]):
                    model_configurations.append(all_batch_files)

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
            print(f"\n--- Training Model v{model_version_num} using {len(files_for_this_model)} batch files ---")

            raw_df_model = spark.read.option("multiLine", "true").json(files_for_this_model)
            if 'ingredients' not in raw_df_model.columns:
                print(f"ERROR: Column 'ingredients' not found for Model v{model_version_num}. Skipping.")
                continue

            data_df_model = raw_df_model.select("title", "ingredients").filter(col("ingredients").isNotNull()) # Pilih kolom yang relevan
            data_df_model = data_df_model.withColumn("cleaned_ingredients", clean_ingredients_udf(col("ingredients")))
            
            # Log sampel data setelah UDF
            print(f"Sample data after UDF for Model v{model_version_num} (first 5 rows):")
            data_df_model.select("ingredients", "cleaned_ingredients").show(5, truncate=False)

            processed_df_model = data_df_model.filter(size(col("cleaned_ingredients")) > 0)
            
            num_data_points = processed_df_model.count()
            print(f"Model v{model_version_num}: Rows after UDF and filtering empty ingredients: {num_data_points}")

            # Minimal data point yang lebih ketat
            min_required_points = K_VALUE_FOR_KMEANS * MIN_DATA_POINTS_FACTOR
            if num_data_points < min_required_points :
                print(f"WARNING: Model v{model_version_num} - Not enough data points ({num_data_points}). "
                      f"Required at least {min_required_points} (K * {MIN_DATA_POINTS_FACTOR}). Skipping this model.")
                continue

            # Definisikan pipeline ML
            # Gunakan stopwords bahasa Inggris default, atau sediakan custom list jika perlu
            # stop_words = StopWordsRemover.loadDefaultStopWords("english") + ["tbsp", "tsp", "cup", "oz", "g", "ml", "lb", "optional", "fresh", "dried"]
            remover = StopWordsRemover(inputCol="cleaned_ingredients", outputCol="filtered_ingredients") # , stopWords=stop_words tambahkan jika mau custom
            
            hashingTF = HashingTF(inputCol="filtered_ingredients", outputCol="raw_features", numFeatures=HASHINGTF_NUM_FEATURES)
            idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=2) # minDocFreq: ignore terms that appear in less than X documents

            # ---- Alternatif Featurization: Word2Vec ----
            # word2Vec = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered_ingredients", outputCol="features")
            # kmeans = KMeans(featuresCol="features", k=K_VALUE_FOR_KMEANS, seed=1, initMode="k-means||", maxIter=20, tol=1e-4)
            # pipeline = Pipeline(stages=[remover, word2Vec, kmeans]) # Jika pakai Word2Vec

            # ---- Pipeline dengan HashingTF + IDF ----
            kmeans = KMeans(featuresCol="features", k=K_VALUE_FOR_KMEANS, seed=1,
                            initMode="k-means||", maxIter=20, tol=1e-4) # Parameter standar
            pipeline = Pipeline(stages=[remover, hashingTF, idf, kmeans])


            # ---- Alternatif Algoritma Clustering: BisectingKMeans ----
            # bkm = BisectingKMeans(featuresCol="features", k=K_VALUE_FOR_KMEANS, seed=1, maxIter=20, minDivisibleClusterSize=1.0)
            # pipeline = Pipeline(stages=[remover, hashingTF, idf, bkm]) # Jika pakai BisectingKMeans

            print(f"Starting training for Model v{model_version_num}...")
            model = pipeline.fit(processed_df_model)
            print(f"Training finished for Model v{model_version_num}.")

            # Evaluasi Model (Silhouette Score)
            predictions_df = model.transform(processed_df_model)
            evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
            silhouette = evaluator.evaluate(predictions_df)
            print(f"Model v{model_version_num}: Silhouette Score = {silhouette:.4f}") # Silhouette: -1 (buruk) hingga 1 (baik), 0 (overlap)

            # Tampilkan distribusi cluster
            print(f"Cluster distribution for Model v{model_version_num}:")
            predictions_df.groupBy("prediction").count().orderBy("prediction").show()
            
            # Jika silhouette sangat rendah atau semua data masuk ke satu cluster, beri peringatan
            cluster_counts = predictions_df.groupBy("prediction").count().collect()
            if silhouette < 0.1 or len(cluster_counts) < K_VALUE_FOR_KMEANS / 2 or any(row['count'] > num_data_points * 0.8 for row in cluster_counts):
                print(f"WARNING: Model v{model_version_num} may have poor clustering quality. Silhouette: {silhouette:.4f}. Check cluster distribution.")


            model_name = f"recipe_cluster_model_v{model_version_num}"
            model_path = os.path.join(MODEL_OUTPUT_PATH, model_name)
            
            if os.path.exists(model_path):
                print(f"Removing existing model directory: {model_path}")
                shutil.rmtree(model_path)
            model.save(model_path)
            print(f"Model {model_name} saved to {model_path}")

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