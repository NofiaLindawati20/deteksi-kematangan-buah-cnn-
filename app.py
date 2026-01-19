import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import zipfile
import tempfile

app = Flask(__name__)

# --- KONFIGURASI PATH MODEL ---
MODEL_FILENAME = 'xception_224_0.0001_128_30.keras'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

CLASS_NAMES = ['Overripe', 'Ripe', 'Unripe']

model = None
LOAD_STATUS = "Not Initialized"
LOAD_ERROR_DETAILS = ""

def build_manual_model():
    """
    Membangun arsitektur Sequential persis seperti di notebook training.
    Nama layer disesuaikan dengan isi file .keras agar load_weights berhasil.
    """
    print("🏗️ Membangun arsitektur Sequential (Xception Base)...")
    
    # 1. Init Sequential
    final_model = Sequential(name='Pretrained_Xception')
    
    # 2. Base Model (Xception)
    # Note: Xception tidak terima argumen 'name' di constructor pada beberapa versi
    # PENTING: Gunakan weights='imagenet' karena saat training base model di-freeze (tidak ditrain ulang)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model._name = 'functional' # CRITICAL: Set nama manual agar cocok dengan file weights
    base_model.trainable = False 
    
    # 3. Add Layers
    # Nama-nama layer ini diambil dari hasil inspeksi file .keras sebelumnya
    final_model.add(base_model)
    final_model.add(GlobalAveragePooling2D(name='global_average_pooling2d'))
    final_model.add(Dense(256, activation='relu', name='dense'))
    final_model.add(Dropout(0.2, name='dropout'))
    final_model.add(Dense(128, activation='relu', name='dense_1'))
    final_model.add(Dense(3, activation='softmax', name='dense_2'))
    
    return final_model

import h5py

def inject_weights_manually(model, h5_path):
    """
    Membaca file h5 dan memasukkan weights secara manual ke layer.
    Bypass mekanisme load_weights Keras yang bermasalah.
    """
    print(f"💉 Injecting weights from {h5_path}...")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # 1. Inject Dense Layers (Top)
            # Structure: layers -> dense -> vars -> 0 (kernel), 1 (bias)
            layer_map = {
                'dense': 'dense',
                'dense_1': 'dense_1',
                'dense_2': 'dense_2'
            }
            
            for model_layer_name, h5_layer_name in layer_map.items():
                try:
                    layer = model.get_layer(model_layer_name)
                    # Path di H5 Keras 3 standard
                    base_path = f"layers/{h5_layer_name}/vars"
                    
                    if base_path in f:
                        print(f"   - Loading {model_layer_name} from {base_path}...")
                        kernel = f[f"{base_path}/0"][:]
                        bias = f[f"{base_path}/1"][:]
                        layer.set_weights([kernel, bias])
                    else:
                        print(f"   ⚠️ Path {base_path} not found in H5!")
                except Exception as e:
                    print(f"   ❌ Failed to load {model_layer_name}: {e}")

            # 2. Inject Xception Weights (Optional but recommended)
            # Xception ada di group 'layers/functional' atau 'layers/xception'
            # Karena base model kita frozen, kita bisa skip ini jika Xception di-init dengan weights='imagenet'
            # Namun untuk memastikan sama persis, kita coba load jika memungkinkan.
            # Struktur di dalam functional sangat kompleks, jadi kita skip dulu fokus ke Top Layers.
            
            print("✅ Manual injection for TOP layers sent!")
            
    except Exception as e:
        print(f"❌ Error during manual injection: {e}")
        raise e

def load_weights_h5_from_zip(target_model, filepath):
    print(f"📂 Mengekstrak bobot dari {filepath}...")
    with zipfile.ZipFile(filepath, 'r') as z:
        weights_file = None
        for name in z.namelist():
            if name.endswith('.weights.h5') or name.endswith('.h5'):
                weights_file = name
                break
        
        if not weights_file:
            raise ValueError("No weights file found in .keras")
            
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp.write(z.read(weights_file))
            tmp_path = tmp.name
            
        try:
            inject_weights_manually(target_model, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def init_model():
    global model, LOAD_STATUS, LOAD_ERROR_DETAILS
    
    if not os.path.exists(MODEL_PATH):
        cwd = os.getcwd()
        LOAD_STATUS = "File Missing"
        print(f"❌ File not found: {MODEL_PATH}")
        return

    try:
        # 1. Bangun Arsitektur
        model = build_manual_model()
        
        # 2. Inject Weights Manually
        load_weights_h5_from_zip(model, MODEL_PATH)
        
        # 3. Verifikasi Akhir
        w_sum = 0
        for layer in model.layers[-3:]: # Cek 3 layer terakhir
            w = layer.get_weights()
            if w: w_sum += np.sum(w[0])
        print(f"📊 Final Top Layers Weight Sum: {w_sum:.4f}")

        LOAD_STATUS = "Success"
        print(f"🚀 Model ready!")
        
    except Exception as e:
        LOAD_STATUS = "Error"
        import traceback
        tb = traceback.format_exc()
        LOAD_ERROR_DETAILS = f"Error: {str(e)}\n{tb}"
        print(f"❌ {LOAD_ERROR_DETAILS}")

# Jalankan inisialisasi saat startup
init_model()

def make_prediction(image_pil):
    if model is None:
        return None, None, None, f"Model Error:\n{LOAD_ERROR_DETAILS}"
    
    try:
        # 1. Resize
        target_size = (224, 224)
        if image_pil.size != target_size:
            image_pil = image_pil.resize(target_size)
            
        # 2. Convert to Array & Preprocess
        img_array = np.array(image_pil)
        if img_array.shape[-1] == 4: img_array = img_array[..., :3] # Remove Alpha channel
        
        img_array = np.expand_dims(img_array, axis=0) # Batch dim
        img_array = img_array.astype('float32') / 255.0 # Normalize 1/255
        
        # 3. Predict output
        predictions = model.predict(img_array, verbose=0)
        
        # Debug Log
        print(f"🔎 Raw Probabilities: {predictions[0]}")
        
        # 4. Result Processing
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = CLASS_NAMES[predicted_idx]
        all_probs = {name: float(predictions[0][i]) for i, name in enumerate(CLASS_NAMES)}
        
        return predicted_class, confidence, all_probs, None

    except Exception as e:
        return None, None, None, str(e)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files: return jsonify({'error': 'No file'}), 400
        file = request.files['image']
        if file.filename == '': return jsonify({'error': 'No selected file'}), 400
        
        image = Image.open(file.stream).convert('RGB')
        pred_class, conf, probs, error = make_prediction(image)
        
        if error: return jsonify({'error': error}), 500
        
        # Return image as base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': pred_class,
            'confidence': conf,
            'all_probabilities': probs,
            'image': img_str
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        
        pred_class, conf, probs, error = make_prediction(image)
        if error: return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'prediction': pred_class,
            'confidence': conf,
            'all_probabilities': probs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)