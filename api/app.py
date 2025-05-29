# api/app.py

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf, col # Tidak perlu regexp_replace atau lower di sini karena UDF sudah menanganinya
from pyspark.sql.types import ArrayType, StringType, StructType, StructField
import os
import json # Untuk UDF
import re   # Untuk UDF

app = Flask(__name__)

# --- Konfigurasi Spark dan Model ---
SPARK_APP_NAME = "RecipeClusterAPI"
SPARK_MASTER = "local[*]" # Jalankan Spark lokal di dalam proses API
MODEL_BASE_PATH = "../models_output/" # Path relatif dari app.py ke folder models_output

# Global Spark Session dan Model yang Dimuat
spark_session = None
loaded_models = {} # Dictionary: {"model_name": model_object}



# --- UDF untuk Preprocessing Input (HARUS SAMA DENGAN SAAT TRAINING) ---
def clean_ingredients_api_udf_logic(ingredients_str_or_list):
    """
    Membersihkan input ingredients.
    Input bisa berupa string tunggal bahan dipisah koma, atau list string bahan.
    Outputnya adalah list of cleaned words.
    """
    if not ingredients_str_or_list:
        return []
    
    all_words = []
    
    # Jika input adalah string tunggal, split dulu jadi list frasa
    # Misal: "onion, garlic, tomato"
    if isinstance(ingredients_str_or_list, str):
        list_of_ingredient_phrases = [phrase.strip() for phrase in ingredients_str_or_list.split(',')]
    elif isinstance(ingredients_str_or_list, list):
        list_of_ingredient_phrases = ingredients_str_or_list
    else:
        return [] # Tipe input tidak dikenal

    for phrase in list_of_ingredient_phrases:
        if not isinstance(phrase, str):
            continue
        cleaned_phrase = phrase.lower()
        cleaned_phrase = re.sub(r"[^a-z\s]", "", cleaned_phrase)
        cleaned_phrase = re.sub(r"\s+", " ", cleaned_phrase).strip()
        words_in_phrase = cleaned_phrase.split()
        for word in words_in_phrase:
            if len(word) > 2:
                all_words.append(word)
    return all_words

# Spark UDF wrapper
# Input ke UDF ini dari DataFrame akan selalu string (kolom 'ingredients_input_string')
clean_ingredients_spark_udf = udf(clean_ingredients_api_udf_logic, ArrayType(StringType()))


def get_spark_session():
    global spark_session
    if spark_session is None:
        print("Initializing SparkSession for API...")
        spark_session = SparkSession.builder \
            .appName(SPARK_APP_NAME) \
            .master(SPARK_MASTER) \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .getOrCreate()
        # Set log level lebih rendah untuk API agar tidak terlalu verbose di konsol API
        spark_session.sparkContext.setLogLevel("ERROR") 
        print("SparkSession initialized.")
    return spark_session

def load_spark_model(model_name_version):
    """Memuat model Spark berdasarkan nama dan versi, misal 'recipe_cluster_model_v2'."""
    if model_name_version not in loaded_models:
        model_full_path = os.path.join(MODEL_BASE_PATH, model_name_version)
        if not os.path.exists(model_full_path):
            print(f"Model path {model_full_path} not found!")
            return None
        try:
            print(f"Loading model from: {model_full_path}")
            # Pastikan SparkSession sudah ada sebelum memuat model
            get_spark_session() 
            model = PipelineModel.load(model_full_path)
            loaded_models[model_name_version] = model
            print(f"Model {model_name_version} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name_version}: {e}")
            return None
    return loaded_models.get(model_name_version)

# --- Pre-load model saat startup API (opsional, tapi baik untuk performa request pertama) ---
# Daftar model yang ingin di-load saat startup. Sesuaikan dengan model yang Anda miliki.
MODELS_TO_PRELOAD = ["recipe_cluster_model_v1", "recipe_cluster_model_v2", "recipe_cluster_model_v3"] 
# Anda bisa juga menambahkan v1 jika ada dan valid.

# @app.before_first_request
def preload_models():
    print("Preloading models...")
    for model_name in MODELS_TO_PRELOAD:
        if os.path.exists(os.path.join(MODEL_BASE_PATH, model_name)): # Cek apakah model ada
            load_spark_model(model_name)
        else:
            print(f"Skipping preload for {model_name}: Directory not found.")
    print("Model preloading finished.")


# --- Endpoints API ---

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to Recipe Clustering API!",
        "available_endpoints": {
            "/predict/v2": "POST with JSON {'ingredients': 'ingredient1, ingredient2, ...'}",
            "/predict/v3": "POST with JSON {'ingredients': 'ingredient1, ingredient2, ...'}",
            "/models_info": "GET information about available models"
        }
    })

def predict_with_model(model_name_version, request_data):
    spark = get_spark_session()
    model = load_spark_model(model_name_version)

    if not model:
        return jsonify({"error": f"Model {model_name_version} is not available or failed to load."}), 500

    if not request_data or 'ingredients' not in request_data:
        return jsonify({"error": "Please provide 'ingredients' in JSON body (e.g., 'onion, garlic, tomato')"}), 400

    ingredients_input_str = request_data['ingredients'] # Ini adalah string, misal "onion, garlic"

    try:
        # Buat DataFrame Spark dari input user
        # Skema input untuk UDF: satu kolom string
        # Nama kolom input untuk pipeline harus sama dengan 'inputCol' dari tahap pertama pipeline (StopWordsRemover)
        # yaitu "cleaned_ingredients". Jadi UDF kita harus menghasilkan kolom bernama "cleaned_ingredients".
        
        # Kita akan buat DataFrame dengan kolom sementara, lalu UDF akan menghasilkan 'cleaned_ingredients'
        # Kolom yang dibaca oleh UDF `clean_ingredients_spark_udf`
        input_schema = StructType([StructField("ingredients_input_string", StringType(), True)])
        input_df = spark.createDataFrame([(ingredients_input_str,)], input_schema)

        # Terapkan UDF untuk membuat kolom "cleaned_ingredients"
        # Kolom "cleaned_ingredients" ini yang akan dipakai oleh StopWordsRemover dalam pipeline
        input_df_processed = input_df.withColumn("cleaned_ingredients", clean_ingredients_spark_udf(col("ingredients_input_string")))
        print(f"DEBUG API - Input string: {ingredients_input_str}") # Tambahkan ini
        print("DEBUG API - DataFrame input_df_processed sebelum transform:") # Tambahkan ini
        input_df_processed.show(truncate=False) # AKTIFKAN INI untuk melihat cleaned_ingredients
        # input_df_processed.show(truncate=False) # Untuk debugging input ke model

        # Lakukan prediksi menggunakan pipeline model yang sudah dimuat
        # Pipeline akan melakukan semua transformasi (StopWordsRemover, HashingTF, IDF) dan prediksi K-Means
        
        
        prediction_df = model.transform(input_df_processed)
        # prediction_df.show(truncate=False) # Untuk debugging output dari model

        print("DEBUG API - DataFrame prediction_df setelah transform:") # Tambahkan ini
        prediction_df.select("cleaned_ingredients", "features", "prediction").show(truncate=False)
        # Ambil hasil prediksi (cluster id)
        cluster_id = prediction_df.select("prediction").first()[0]
        
        return jsonify({
            "model_version_used": model_name_version,
            "input_ingredients_string": ingredients_input_str,
            "predicted_cluster": int(cluster_id)
        })

    except Exception as e:
        print(f"Error during prediction with {model_name_version}: {e}")
        import traceback
        traceback.print_exc() # Cetak traceback untuk debug lebih detail
        return jsonify({"error": f"Prediction error with {model_name_version}: {str(e)}"}), 500


@app.route('/predict/v2', methods=['POST'])
def predict_v2_endpoint():
    """Endpoint untuk prediksi menggunakan model v2."""
    data = request.get_json()
    return predict_with_model("recipe_cluster_model_v2", data)

@app.route('/predict/v3', methods=['POST'])
def predict_v3_endpoint():
    """Endpoint untuk prediksi menggunakan model v3."""
    data = request.get_json()
    return predict_with_model("recipe_cluster_model_v3", data)

# Endpoint ketiga: Informasi model
@app.route('/models_info', methods=['GET'])
def get_models_information():
    """Memberikan informasi tentang model yang tersedia dan dimuat."""
    available_on_disk = []
    if os.path.exists(MODEL_BASE_PATH):
        for item_name in os.listdir(MODEL_BASE_PATH):
            item_path = os.path.join(MODEL_BASE_PATH, item_name)
            # Cek apakah itu direktori dan namanya sesuai pola model kita
            if os.path.isdir(item_path) and item_name.startswith("recipe_cluster_model_v"):
                available_on_disk.append(item_name)
    
    return jsonify({
        "message": "Model Information",
        "models_available_on_disk": sorted(available_on_disk),
        "models_loaded_in_memory": sorted(list(loaded_models.keys()))
    })


if __name__ == '__main__':
    # get_spark_session() # Inisialisasi SparkSession saat startup (opsional, preload_models akan melakukannya)
    preload_models() # Panggil secara eksplisit jika @app.before_first_request tidak selalu dipanggil di dev server
    
    print("Starting Flask API server...")
    # Pastikan host='0.0.0.0' agar bisa diakses dari luar container jika API ini di-Dockerize nanti
    # port 5000 adalah default Flask
    app.run(host='0.0.0.0', port=5001, debug=True) # Gunakan debug=True untuk development