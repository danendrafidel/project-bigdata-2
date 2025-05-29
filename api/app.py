# api/app.py

import os
import sys

# ---------------------------------------------------------------------------
# PENTING: Atur environment variable PYSPARK_PYTHON dan PYSPARK_DRIVER_PYTHON
# sebelum mengimpor modul pyspark apa pun.
# ---------------------------------------------------------------------------
if 'PYSPARK_PYTHON' not in os.environ:
    os.environ['PYSPARK_PYTHON'] = sys.executable
if 'PYSPARK_DRIVER_PYTHON' not in os.environ:
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, StructType, StructField
import json # Untuk UDF
import re   # Untuk UDF

app = Flask(__name__)

# --- Konfigurasi Spark dan Model ---
SPARK_APP_NAME = "RecipeClusterAPI"
SPARK_MASTER = "local[*]"

# Path absolut ke direktori skrip api/app.py
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Path ke models_output relatif dari direktori proyek (satu tingkat di atas 'api')
MODEL_BASE_PATH = os.path.join(_current_script_dir, "..", "models_output")
print(f"DEBUG: Calculated MODEL_BASE_PATH: {MODEL_BASE_PATH}")


# Global Spark Session dan Model yang Dimuat
spark_session = None
loaded_models = {} # Dictionary: {"model_name": model_object}


# --- UDF untuk Preprocessing Input (HARUS SAMA DENGAN SAAT TRAINING) ---
def clean_ingredients_api_udf_logic(ingredients_str_or_list):
    if not ingredients_str_or_list:
        return []
    
    all_words = []
    
    if isinstance(ingredients_str_or_list, str):
        list_of_ingredient_phrases = [phrase.strip() for phrase in ingredients_str_or_list.split(',')]
    elif isinstance(ingredients_str_or_list, list):
        list_of_ingredient_phrases = ingredients_str_or_list
    else:
        return []

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
        spark_session.sparkContext.setLogLevel("ERROR") 
        print("SparkSession initialized.")
    return spark_session

def load_spark_model(model_name_version):
    global loaded_models # Pastikan kita memodifikasi global dict
    if model_name_version not in loaded_models:
        model_full_path = os.path.join(MODEL_BASE_PATH, model_name_version)
        print(f"Attempting to load model from: {model_full_path}") # Debug path
        if not os.path.exists(model_full_path) or not os.path.isdir(model_full_path):
            print(f"Model path {model_full_path} not found or is not a directory!")
            return None
        try:
            get_spark_session() 
            model = PipelineModel.load(model_full_path)
            loaded_models[model_name_version] = model
            print(f"Model {model_name_version} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name_version}: {e}")
            import traceback
            traceback.print_exc()
            return None
    return loaded_models.get(model_name_version)

MODELS_TO_PRELOAD = ["recipe_cluster_model_v1", "recipe_cluster_model_v2", "recipe_cluster_model_v3"] 

def preload_models_on_startup(): # Ubah nama fungsi agar lebih jelas
    print("Preloading models...")
    for model_name in MODELS_TO_PRELOAD:
        model_dir_path = os.path.join(MODEL_BASE_PATH, model_name)
        if os.path.exists(model_dir_path) and os.path.isdir(model_dir_path):
            load_spark_model(model_name)
        else:
            print(f"Skipping preload for {model_name}: Directory {model_dir_path} not found.")
    print("Model preloading finished.")

@app.route('/', methods=['GET'])
def home():
    # Dapatkan model yang tersedia dari direktori
    available_on_disk_now = []
    if os.path.exists(MODEL_BASE_PATH) and os.path.isdir(MODEL_BASE_PATH):
        for item_name in os.listdir(MODEL_BASE_PATH):
            item_path = os.path.join(MODEL_BASE_PATH, item_name)
            if os.path.isdir(item_path) and item_name.startswith("recipe_cluster_model_v"):
                available_on_disk_now.append(item_name)
    
    return jsonify({
        "message": "Welcome to Recipe Clustering API!",
        "model_base_path_used": MODEL_BASE_PATH, # Tampilkan path yang digunakan
        "models_available_on_disk": sorted(available_on_disk_now),
        "models_loaded_in_memory": sorted(list(loaded_models.keys())),
        "available_endpoints": {
            "/predict/v<N>": "POST with JSON {'ingredients': 'ingredient1, ingredient2, ...'} (replace <N> with model version)",
            "/models_info": "GET information about available models"
        }
    })

def predict_with_model(model_name_version, request_data):
    spark = get_spark_session() # Pastikan SparkSession ada
    model = load_spark_model(model_name_version)

    if not model:
        return jsonify({"error": f"Model {model_name_version} is not available or failed to load."}), 500

    if not request_data or 'ingredients' not in request_data:
        return jsonify({"error": "Please provide 'ingredients' in JSON body (e.g., {'ingredients': 'onion, garlic, tomato'})"}), 400

    ingredients_input_str = request_data['ingredients']

    try:
        # Input untuk UDF adalah string tunggal dari request
        # UDF akan menghasilkan kolom 'cleaned_ingredients'
        input_schema = StructType([StructField("ingredients_input_string", StringType(), True)])
        input_df = spark.createDataFrame([(ingredients_input_str,)], schema=input_schema)

        # Terapkan UDF. Nama output kolom dari UDF harus "cleaned_ingredients"
        # karena itu adalah inputCol untuk StopWordsRemover di pipeline Anda.
        input_df_processed = input_df.withColumn("cleaned_ingredients", clean_ingredients_spark_udf(col("ingredients_input_string")))
        
        print(f"\nDEBUG API - Model: {model_name_version}, Input string: '{ingredients_input_str}'")
        print("DEBUG API - DataFrame input_df_processed (input to model.transform):")
        input_df_processed.printSchema()
        input_df_processed.show(truncate=False)
        
        prediction_df = model.transform(input_df_processed)
        
        print("DEBUG API - DataFrame prediction_df (output from model.transform):")
        prediction_df.printSchema()
        prediction_df.select("ingredients_input_string", "cleaned_ingredients", "filtered_ingredients", "raw_features", "features", "prediction").show(truncate=False)
        
        cluster_id = prediction_df.select("prediction").first()[0]
        
        return jsonify({
            "model_version_used": model_name_version,
            "input_ingredients_string": ingredients_input_str,
            "predicted_cluster": int(cluster_id)
        })

    except Exception as e:
        print(f"Error during prediction with {model_name_version}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction error with {model_name_version}: {str(e)}"}), 500

# --- Dinamis membuat endpoint predict berdasarkan model yang ada ---
def create_predict_endpoints():
    if not os.path.exists(MODEL_BASE_PATH) or not os.path.isdir(MODEL_BASE_PATH):
        print(f"WARNING: Model directory {MODEL_BASE_PATH} not found. No dynamic predict endpoints will be created.")
        return

    for model_folder_name in os.listdir(MODEL_BASE_PATH):
        model_folder_path = os.path.join(MODEL_BASE_PATH, model_folder_name)
        if os.path.isdir(model_folder_path) and model_folder_name.startswith("recipe_cluster_model_v"):
            version_str = model_folder_name.replace("recipe_cluster_model_v", "")
            try:
                version_num = int(version_str) # Pastikan itu angka
                endpoint_path = f'/predict/v{version_num}'
                
                # Buat fungsi endpoint secara dinamis
                def create_dynamic_endpoint_func(model_name):
                    def dynamic_endpoint_func():
                        data = request.get_json()
                        return predict_with_model(model_name, data)
                    return dynamic_endpoint_func

                # Beri nama unik untuk fungsi agar Flask bisa mendaftarkannya
                endpoint_func_name = f"predict_endpoint_v{version_num}"
                dynamic_func = create_dynamic_endpoint_func(model_folder_name)
                dynamic_func.__name__ = endpoint_func_name # Penting untuk Flask

                app.add_url_rule(endpoint_path, view_func=dynamic_func, methods=['POST'])
                print(f"Dynamically created endpoint: POST {endpoint_path} for model {model_folder_name}")

            except ValueError:
                print(f"Could not parse version number from model folder: {model_folder_name}")

@app.route('/models_info', methods=['GET'])
def get_models_information():
    available_on_disk = []
    if os.path.exists(MODEL_BASE_PATH) and os.path.isdir(MODEL_BASE_PATH):
        for item_name in os.listdir(MODEL_BASE_PATH):
            item_path = os.path.join(MODEL_BASE_PATH, item_name)
            if os.path.isdir(item_path) and item_name.startswith("recipe_cluster_model_v"):
                available_on_disk.append(item_name)
    
    return jsonify({
        "message": "Model Information",
        "model_base_path_used": MODEL_BASE_PATH,
        "models_available_on_disk": sorted(available_on_disk),
        "models_loaded_in_memory": sorted(list(loaded_models.keys()))
    })


if __name__ == '__main__':
    # Inisialisasi Spark dan pre-load model sebelum Flask app.run()
    # Ini penting agar tidak dilakukan di dalam request pertama yang bisa lambat
    get_spark_session() # Panggil sekali untuk inisialisasi
    create_predict_endpoints() # Buat endpoint predict berdasarkan model di disk
    preload_models_on_startup() # Preload model yang didefinisikan
    
    print("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    # use_reloader=False penting saat debug Spark agar SparkContext tidak dibuat ulang terus-menerus
    # yang bisa menyebabkan error. Matikan reloader jika ada masalah Spark context saat debug.
    # Untuk produksi, debug=False dan reloader tidak akan aktif.