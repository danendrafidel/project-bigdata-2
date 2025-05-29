# model/train_model.py

import os
import sys
import glob
import json
import re
import shutil

# ... (PYSPARK_PYTHON setup) ...
if 'PYSPARK_PYTHON' not in os.environ:
    os.environ['PYSPARK_PYTHON'] = sys.executable
if 'PYSPARK_DRIVER_PYTHON' not in os.environ:
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Word2Vec, CountVectorizer
from pyspark.ml.clustering import KMeans, BisectingKMeans # GaussianMixtureModel (GMM) bisa ditambahkan jika relevan
# from pyspark.ml.clustering import LDA # Jika ingin eksplor LDA
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col, size
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.evaluation import ClusteringEvaluator

# Konfigurasi Global
SPARK_APP_NAME = "RecipeClusteringMultiAlgo"
SPARK_MASTER = "local[*]"
BATCH_DATA_PATH = "dataset/batch_output/" # Atau path ke dataset besar Anda
MODEL_OUTPUT_PATH = "models_output/"

K_DEFAULT = 10 # Nilai K default, bisa di-override per model
HASHINGTF_NUM_FEATURES_DEFAULT = 8000
IDF_MIN_DOC_FREQ_DEFAULT = 5
WORD2VEC_VECTOR_SIZE_DEFAULT = 100
WORD2VEC_MIN_COUNT_DEFAULT = 5

# Pastikan direktori output model ada
if not os.path.exists(MODEL_OUTPUT_PATH):
    os.makedirs(MODEL_OUTPUT_PATH)
    print(f"Created directory: {MODEL_OUTPUT_PATH}")

def create_spark_session():
    print(f"Attempting to create SparkSession with PYSPARK_PYTHON='{os.environ.get('PYSPARK_PYTHON')}'")
    return SparkSession.builder.appName(SPARK_APP_NAME)\
        .master(SPARK_MASTER)\
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()

# --- UDF clean_ner_ingredients (sama seperti sebelumnya, fokus pada kolom NER) ---
def clean_ner_ingredients(ner_column_value):
    if not ner_column_value: return []
    ingredients_list = []
    if isinstance(ner_column_value, str):
        ingredients_list = [item.strip().lower() for item in ner_column_value.split(',') if item.strip()]
    elif isinstance(ner_column_value, list):
        ingredients_list = [str(item).strip().lower() for item in ner_column_value if isinstance(item, (str, int, float)) and str(item).strip()]
    else: return []
    cleaned_words = []
    for phrase in ingredients_list:
        phrase = re.sub(r'\([^)]*\)', '', phrase)
        phrase = re.sub(r'\b(oz|ounce|g|gram|kg|lb|pound|cup|tbsp|tsp|ml|l|to taste|as needed|chopped|sliced|diced|minced|peeled|fresh|dried|ground|optional)\b', '', phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"[^a-z0-9\s-]", "", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        words_in_phrase = phrase.split()
        for word in words_in_phrase:
            if len(word) > 2 and word not in ['and', 'the', 'for', 'with', 'into', 'onto', 'from', 'or', 'some', 'other', 'type', 'brand']:
                cleaned_words.append(word)
    return list(set(cleaned_words))

clean_ner_udf = udf(clean_ner_ingredients, ArrayType(StringType()))

if __name__ == "__main__":
    spark = None
    try:
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("WARN")
        print("SparkSession created successfully.")

        # Membaca semua data terlebih dahulu
        json_files_glob_pattern = os.path.join(BATCH_DATA_PATH, "*.json")
        all_batch_files = sorted(glob.glob(json_files_glob_pattern))
        if not all_batch_files:
            print(f"No data files found. Exiting.")
            if spark: spark.stop(); exit()
        
        print(f"Reading {len(all_batch_files)} batch files...")
        raw_df_full = spark.read.option("multiLine", "true").json(all_batch_files)
        raw_df_full.cache()
        total_rows = raw_df_full.count()
        print(f"Total rows in dataset: {total_rows}")

        if 'NER' not in raw_df_full.columns:
            print(f"ERROR: Column 'NER' not found. Exiting.")
            # Perbaikan untuk blok if ini juga:
            if spark:
                spark.stop()
            exit() # exit() harus di luar if spark, tapi di dalam if 'NER' not in ...

        data_df_processed_full = raw_df_full.select("title", "link", "source", "site", "NER") \
            .filter(col("NER").isNotNull()) \
            .withColumn("cleaned_ner_ingredients", clean_ner_udf(col("NER"))) \
            .filter(size(col("cleaned_ner_ingredients")) > 0)
        
        data_df_processed_full.cache()
        processed_rows = data_df_processed_full.count()
        print(f"Rows after NER cleaning and filtering: {processed_rows}")
        
        if processed_rows == 0:
            print("No data left. Exiting.")
            if spark:
                spark.stop()
            exit() # Pindahkan exit() ke barisnya sendiri

        print("Sample data after NER UDF (first 3 rows):")
        data_df_processed_full.select("NER", "cleaned_ner_ingredients").show(3, truncate=70)

        # Tentukan fraksi data untuk setiap model
        fractions = [round(1/3, 2), round(2/3, 2), 1.0]
        
        # Definisikan konfigurasi untuk 3 model dengan algoritma berbeda
        model_definitions = [
            {"id": "v1", "algo": "kmeans", "feature_method": "tfidf", "k": K_DEFAULT, "data_fraction_idx": 0},
            {"id": "v2", "algo": "bkm", "feature_method": "tfidf", "k": K_DEFAULT, "data_fraction_idx": 1},
            {"id": "v3", "algo": "kmeans", "feature_method": "word2vec", "k": K_DEFAULT, "data_fraction_idx": 2}
        ]
        
        # Jika ingin data kumulatif dari file:
        all_files_count = len(all_batch_files)
        file_splits_paths = [
            all_batch_files[:max(1, all_files_count // 3)],
            all_batch_files[:max(1, (2 * all_files_count) // 3)],
            all_batch_files
        ]


        for i, model_def in enumerate(model_definitions):
            model_name_suffix = f"{model_def['id']}_{model_def['algo']}_{model_def['feature_method']}_k{model_def['k']}"
            model_id_name = f"recipe_cluster_model_{model_name_suffix}"
            
            # Ambil data sesuai fraksi yang ditentukan oleh file_splits_paths
            current_files_for_model = file_splits_paths[model_def['data_fraction_idx']]
            if not current_files_for_model:
                print(f"No files for model {model_id_name}. Skipping."); continue

            print(f"\n--- Training {model_id_name} using {len(current_files_for_model)} batch files ---")
            
            print(f"Reading data for {model_id_name}...")
            current_raw_df = spark.read.option("multiLine", "true").json(current_files_for_model)
            if 'NER' not in current_raw_df.columns:
                print(f"NER column missing for {model_id_name}. Skipping."); continue
            
            current_training_df = current_raw_df.select("title", "NER") \
                .filter(col("NER").isNotNull()) \
                .withColumn("cleaned_ner_ingredients", clean_ner_udf(col("NER"))) \
                .filter(size(col("cleaned_ner_ingredients")) > 0)
            
            current_training_df.cache() # Cache data spesifik model ini
            num_data_points = current_training_df.count()
            print(f"{model_id_name}: Usable data points = {num_data_points}")

            min_required_points = model_def['k'] * 2 # Faktor MIN_DATA_POINTS_FACTOR
            if num_data_points < min_required_points:
                print(f"WARNING: {model_id_name} - Not enough data ({num_data_points}). Req: {min_required_points}. Skipping.");
                current_training_df.unpersist()
                continue

            stages = []
            remover = StopWordsRemover(inputCol="cleaned_ner_ingredients", outputCol="filtered_ingredients")
            stages.append(remover)

            if model_def['feature_method'] == "tfidf":
                hashingTF = HashingTF(inputCol="filtered_ingredients", outputCol="raw_features", numFeatures=HASHINGTF_NUM_FEATURES_DEFAULT)
                idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=IDF_MIN_DOC_FREQ_DEFAULT)
                stages.extend([hashingTF, idf])
            elif model_def['feature_method'] == "word2vec":
                word2Vec = Word2Vec(vectorSize=WORD2VEC_VECTOR_SIZE_DEFAULT, minCount=WORD2VEC_MIN_COUNT_DEFAULT,
                                    inputCol="filtered_ingredients", outputCol="features", seed=42)
                stages.append(word2Vec)
            
            clusterer = None
            if model_def['algo'] == "kmeans":
                clusterer = KMeans(featuresCol="features", k=model_def['k'], seed=1, initMode="k-means||")
            elif model_def['algo'] == "bkm": # BisectingKMeans
                clusterer = BisectingKMeans(featuresCol="features", k=model_def['k'], seed=1, minDivisibleClusterSize=1.0)
            # Tambahkan elif untuk GMM atau LDA jika ingin
            
            if not clusterer:
                print(f"Unknown clustering algo for {model_id_name}. Skipping.")
                current_training_df.unpersist()
                continue
            stages.append(clusterer)
            
            pipeline = Pipeline(stages=stages)
            
            print(f"Starting training for {model_id_name}...")
            try:
                model = pipeline.fit(current_training_df)
                print(f"Training finished for {model_id_name}.")

                predictions_df = model.transform(current_training_df)
                evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="features",
                                                metricName="silhouette", distanceMeasure="squaredEuclidean")
                silhouette = evaluator.evaluate(predictions_df)
                print(f"{model_id_name}: Silhouette Score = {silhouette:.4f}")

                print(f"Cluster distribution for {model_id_name}:")
                predictions_df.groupBy("prediction").count().orderBy(col("count").desc()).show(model_def['k'] + 5)

                model_path = os.path.join(MODEL_OUTPUT_PATH, model_id_name)
                if os.path.exists(model_path):
                    print(f"Removing existing model directory: {model_path}")
                    shutil.rmtree(model_path)
                model.save(model_path)
                print(f"Model {model_id_name} saved to {model_path}")

            except Exception as e_train:
                print(f"ERROR during training or saving {model_id_name}: {e_train}")
                import traceback
                traceback.print_exc()
            finally:
                current_training_df.unpersist() # Penting untuk unpersist setelah selesai dengan data ini

        # Unpersist DataFrame utama jika sudah selesai semua
        data_df_processed_full.unpersist()
        raw_df_full.unpersist()

    except Exception as e_main:
        print(f"An critical error occurred: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping SparkSession...")
            spark.stop()
            print("SparkSession stopped.")
        print("All model training attempts finished.")