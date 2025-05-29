# api/app.py

import os
import sys

# ... (PYSPARK_PYTHON setup) ...
if 'PYSPARK_PYTHON' not in os.environ:
    os.environ['PYSPARK_PYTHON'] = sys.executable
if 'PYSPARK_DRIVER_PYTHON' not in os.environ:
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, StructType, StructField
import json
import re

app = Flask(__name__)

SPARK_APP_NAME = "RecipeClusterAPI"
SPARK_MASTER = "local[*]"

_current_script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE_PATH = os.path.join(_current_script_dir, "..", "models_output")
print(f"DEBUG: Calculated MODEL_BASE_PATH: {MODEL_BASE_PATH}")

spark_session = None
loaded_models = {}

# Jika Anda ingin API menerima format yang sama dengan kolom NER di training, UDF ini perlu disesuaikan.
def clean_api_input_ingredients(ingredients_str_or_list): # Diganti nama agar lebih jelas
    if not ingredients_str_or_list: return []
    all_words = []
    if isinstance(ingredients_str_or_list, str):
        # Ini adalah input yang diharapkan dari API: "bawang, tomat, ayam"
        list_of_ingredient_phrases = [phrase.strip().lower() for phrase in ingredients_str_or_list.split(',')]
    elif isinstance(ingredients_str_or_list, list): # Jika API dikirimi list
        list_of_ingredient_phrases = [str(item).strip().lower() for item in ingredients_str_or_list]
    else: return []

    processed_words = []
    for phrase in list_of_ingredient_phrases:
        # Pembersihan standar, mirip dengan yang di train_model.py untuk NER tapi mungkin lebih sederhana
        # karena input API mungkin tidak sekotor data mentah.
        # Sesuaikan ini agar cocok dengan bagaimana Anda ingin input API diproses
        # agar konsisten dengan input ke pipeline model Anda.
        phrase = re.sub(r'\([^)]*\)', '', phrase)
        phrase = re.sub(r'\b(oz|ounce|g|gram|kg|lb|pound|cup|tbsp|tsp|ml|l|to taste|as needed|chopped|sliced|diced|minced|peeled|fresh|dried|ground|optional)\b', '', phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"[^a-z0-9\s-]", "", phrase) # Izinkan angka dan hyphen
        phrase = re.sub(r"\s+", " ", phrase).strip()
        words_in_phrase = phrase.split()
        for word in words_in_phrase:
            if len(word) > 2 and word not in ['and', 'the', 'for', 'with', 'into', 'onto', 'from', 'or', 'some', 'other', 'type', 'brand']:
                processed_words.append(word)
    return list(set(processed_words)) # Kata unik

# Nama kolom input ke UDF ini dari DataFrame akan 'api_input_ingredients_string'
api_ingredients_udf = udf(clean_api_input_ingredients, ArrayType(StringType()))


def get_spark_session():
    global spark_session
    if spark_session is None:
        print("Initializing SparkSession for API...")
        spark_session = (SparkSession.builder
                         .appName(SPARK_APP_NAME)
                         .master(SPARK_MASTER)
                         .config("spark.driver.memory", "2g")  # Sesuaikan jika perlu
                         .config("spark.executor.memory", "2g")  # Sesuaikan jika perlu
                         .getOrCreate())
        spark_session.sparkContext.setLogLevel("ERROR") 
        print("SparkSession initialized.")
    return spark_session

def load_spark_model(model_folder_name): # Parameter adalah nama folder lengkap
    global loaded_models
    if model_folder_name not in loaded_models:
        model_full_path = os.path.join(MODEL_BASE_PATH, model_folder_name)
        print(f"Attempting to load model from: {model_full_path}")
        if not os.path.exists(model_full_path) or not os.path.isdir(model_full_path):
            print(f"Model path {model_full_path} not found or is not a directory!")
            return None
        try:
            get_spark_session() 
            model = PipelineModel.load(model_full_path)
            loaded_models[model_folder_name] = model
            print(f"Model {model_folder_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_folder_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    return loaded_models.get(model_folder_name)

# Sesuaikan dengan nama folder model yang benar-benar ada di models_output/
MODELS_TO_PRELOAD = [
    "recipe_cluster_model_v1_kmeans_tfidf_k10",
    "recipe_cluster_model_v2_bkm_tfidf_k10",
    "recipe_cluster_model_v3_kmeans_word2vec_k10"
] 

def preload_models_on_startup():
    print("Preloading models...")
    for model_folder_name in MODELS_TO_PRELOAD: # Gunakan nama folder lengkap
        model_dir_path = os.path.join(MODEL_BASE_PATH, model_folder_name)
        if os.path.exists(model_dir_path) and os.path.isdir(model_dir_path):
            load_spark_model(model_folder_name) # Kirim nama folder lengkap
        else:
            print(f"Skipping preload for {model_folder_name}: Directory {model_dir_path} not found.")
    print("Model preloading finished.")

@app.route('/', methods=['GET'])
def home():
    available_on_disk_now = []
    if os.path.exists(MODEL_BASE_PATH) and os.path.isdir(MODEL_BASE_PATH):
        for item_name in os.listdir(MODEL_BASE_PATH):
            item_path = os.path.join(MODEL_BASE_PATH, item_name)
            # Cek apakah itu direktori dan namanya *dimulai* dengan pola model kita
            if os.path.isdir(item_path) and item_name.startswith("recipe_cluster_model_"):
                available_on_disk_now.append(item_name)
    
    return jsonify({
        "message": "Welcome to Recipe Clustering API!",
        "model_base_path_used": MODEL_BASE_PATH,
        "models_available_on_disk": sorted(available_on_disk_now),
        "models_loaded_in_memory": sorted(list(loaded_models.keys())),
        "available_endpoints_info": "Endpoints are dynamically created, e.g., /predict/v1_kmeans_tfidf_k10. Check /models_info for loaded models."
    })

def predict_with_model(model_folder_name, request_data): # Parameter adalah nama folder lengkap
    spark = get_spark_session()
    model = load_spark_model(model_folder_name) # Muat berdasarkan nama folder lengkap

    if not model:
        return jsonify({"error": f"Model {model_folder_name} is not available or failed to load."}), 500

    if not request_data or 'ingredients' not in request_data:
        return jsonify({"error": "Please provide 'ingredients' (string, comma-separated) in JSON body"}), 400

    ingredients_input_str = request_data['ingredients']

    try:
        # Skema input DataFrame ke UDF
        input_schema = StructType([StructField("api_input_ingredients_string", StringType(), True)])
        input_df = spark.createDataFrame([(ingredients_input_str,)], schema=input_schema)
        
        # UDF API akan menghasilkan kolom bernama `cleaned_ner_ingredients` yang akan digunakan oleh StopWordsRemover
        input_df_processed = input_df.withColumn("cleaned_ner_ingredients", api_ingredients_udf(col("api_input_ingredients_string")))
        
        print(f"\nDEBUG API - Model: {model_folder_name}, Input string: '{ingredients_input_str}'")
        print("DEBUG API - DataFrame input_df_processed (input to model.transform):")
        input_df_processed.printSchema()
        input_df_processed.show(truncate=False)
        
        prediction_df = model.transform(input_df_processed)
        
        print("DEBUG API - DataFrame prediction_df (output from model.transform):")
        prediction_df.printSchema()
        # Sesuaikan kolom yang ditampilkan berdasarkan output pipeline Anda
        # Minimal harus ada "prediction" dan "features". Kolom antara bisa berbeda tergantung pipeline.
        cols_to_show = ["api_input_ingredients_string", "cleaned_ner_ingredients", "features", "prediction"]
        existing_cols_to_show = [c for c in cols_to_show if c in prediction_df.columns]
        if "filtered_ingredients" in prediction_df.columns: existing_cols_to_show.insert(2, "filtered_ingredients")
        if "raw_features" in prediction_df.columns: existing_cols_to_show.insert(3, "raw_features")

        prediction_df.select(existing_cols_to_show).show(truncate=False)
        
        cluster_id = prediction_df.select("prediction").first()[0]
        
        return jsonify({
            "model_used": model_folder_name,
            "input_ingredients_string": ingredients_input_str,
            "predicted_cluster": int(cluster_id)
        })

    except Exception as e:
        print(f"Error during prediction with {model_folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction error with {model_folder_name}: {str(e)}"}), 500

def create_predict_endpoints():
    if not os.path.exists(MODEL_BASE_PATH) or not os.path.isdir(MODEL_BASE_PATH):
        print(f"WARNING: Model directory {MODEL_BASE_PATH} not found. No dynamic predict endpoints.")
        return

    for model_folder_name in os.listdir(MODEL_BASE_PATH):
        model_folder_path = os.path.join(MODEL_BASE_PATH, model_folder_name)
        # Pastikan itu direktori dan namanya valid sebagai model
        if os.path.isdir(model_folder_path) and model_folder_name.startswith("recipe_cluster_model_"):
            # Gunakan bagian setelah "recipe_cluster_model_" sebagai ID endpoint
            endpoint_id = model_folder_name.replace("recipe_cluster_model_", "", 1) # Hapus prefix sekali
            
            if not endpoint_id: # Jika setelah replace jadi string kosong
                print(f"Could not derive endpoint ID from folder: {model_folder_name}")
                continue

            endpoint_path = f'/predict/{endpoint_id}'
            
            def create_dynamic_endpoint_func(folder_name_for_closure):
                def dynamic_endpoint_func():
                    data = request.get_json()
                    return predict_with_model(folder_name_for_closure, data)
                return dynamic_endpoint_func

            endpoint_func_name = f"predict_endpoint_{endpoint_id.replace('-', '_').replace('.', '_')}" # Buat nama fungsi valid
            dynamic_func = create_dynamic_endpoint_func(model_folder_name) # Kirim nama folder lengkap
            dynamic_func.__name__ = endpoint_func_name

            app.add_url_rule(endpoint_path, view_func=dynamic_func, methods=['POST'])
            print(f"Dynamically created endpoint: POST {endpoint_path} for model {model_folder_name}")

@app.route('/models_info', methods=['GET'])
def get_models_information():
    available_on_disk = []
    if os.path.exists(MODEL_BASE_PATH) and os.path.isdir(MODEL_BASE_PATH):
        for item_name in os.listdir(MODEL_BASE_PATH):
            item_path = os.path.join(MODEL_BASE_PATH, item_name)
            if os.path.isdir(item_path) and item_name.startswith("recipe_cluster_model_"):
                available_on_disk.append(item_name)
    
    return jsonify({
        "message": "Model Information",
        "model_base_path_used": MODEL_BASE_PATH,
        "models_available_on_disk": sorted(available_on_disk),
        "models_loaded_in_memory": sorted(list(loaded_models.keys()))
    })

if __name__ == '__main__':
    get_spark_session()
    create_predict_endpoints() # Ini akan membaca folder model dan membuat endpoint
    preload_models_on_startup() # Ini akan mencoba memuat model dari MODELS_TO_PRELOAD
    
    print("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)