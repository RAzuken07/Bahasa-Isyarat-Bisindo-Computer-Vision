import os
import sys
import json
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import MODEL_DIR, SEQUENCE_LENGTH, STATIC_DATA_DIR, DYNAMIC_DATA_DIR
from src.post_processing import PredictionSmoother
from src.preprocessing import normalize_landmarks

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="Computer Vision",
    version="1.2",
    description="API untuk prediksi bahasa isyarat dengan penstabil visual (PredictionSmoother)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# THREAD EXECUTOR (agar model.predict tidak blocking)
# =====================================================
executor = ThreadPoolExecutor(max_workers=2)

# =====================================================
# MODEL LOADING & INITIALIZATION
# =====================================================
STATIC_MODEL_PATH = os.path.join(MODEL_DIR, "static_model.h5")
DYNAMIC_MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_model.h5")
LABEL_STATIC_PATH = os.path.join(MODEL_DIR, "label_static.json")
LABEL_DYNAMIC_PATH = os.path.join(MODEL_DIR, "label_dynamic.json")

static_model = None
dynamic_model = None
labels_static: Dict[int, str] = {}
labels_dynamic: Dict[int, str] = {}

# Load Model Statis
if os.path.exists(STATIC_MODEL_PATH) and os.path.exists(LABEL_STATIC_PATH):
    try:
        static_model = tf.keras.models.load_model(STATIC_MODEL_PATH)
        with open(LABEL_STATIC_PATH, "r") as f:
            data = json.load(f)
            labels_static = {int(k): v for k, v in data.items()}
        print("[OK] Model Statis & Label Berhasil Dimuat.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat Model Statis: {e}")

# Load Model Dinamis
if os.path.exists(DYNAMIC_MODEL_PATH) and os.path.exists(LABEL_DYNAMIC_PATH):
    try:
        dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH)
        with open(LABEL_DYNAMIC_PATH, "r") as f:
            data = json.load(f)
            labels_dynamic = {int(k): v for k, v in data.items()}
        print("[OK] Model Dinamis & Label Berhasil Dimuat.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat Model Dinamis: {e}")

# PERBAIKAN: Menggunakan positional arguments agar aman dari perbedaan nama parameter internal
smoother = PredictionSmoother()

# =====================================================
# TRAINING BACKGROUND TASK STATE
# =====================================================
class TrainingTask:
    def __init__(self):
        self.process = None
        self.status = "idle"  # idle, training, success, failed
        self.logs = []
        self.mode = None

training_task = TrainingTask()

async def run_training_subprocess(mode: str):
    global training_task
    training_task.status = "training"
    training_task.mode = mode
    training_task.logs = [f"Starting training for {mode} model..."]
    
    script_name = "train_static.py" if mode == "STATIC" else "train_dynamic.py"
    script_path = os.path.join(BASE_DIR, script_name)
    
    try:
        # Menjalankan subprocess Python secara asinkron
        cmd = [sys.executable, script_path]
        training_task.logs.append(f"Command: {' '.join(cmd)}")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=BASE_DIR
        )
        training_task.process = proc
        
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            decoded_line = line.decode('utf-8', errors='ignore').strip()
            if decoded_line:
                training_task.logs.append(decoded_line)
                # Batasi log agar tidak membengkak di memori
                if len(training_task.logs) > 500:
                    training_task.logs.pop(1)
        
        await proc.wait()
        if proc.returncode == 0:
            training_task.status = "success"
            training_task.logs.append("[OK] Training completed successfully!")
            reload_models()
        else:
            training_task.status = "failed"
            training_task.logs.append(f"[ERROR] Training failed with return code {proc.returncode}")
            
    except Exception as e:
        training_task.status = "failed"
        training_task.logs.append(f"[ERROR] Error during training: {str(e)}")
        traceback.print_exc()

def reload_models():
    global static_model, dynamic_model, labels_static, labels_dynamic
    print("[INFO] Memuat ulang model AI...")
    
    # Reload static model
    if os.path.exists(STATIC_MODEL_PATH) and os.path.exists(LABEL_STATIC_PATH):
        try:
            static_model = tf.keras.models.load_model(STATIC_MODEL_PATH)
            with open(LABEL_STATIC_PATH, "r") as f:
                data = json.load(f)
                labels_static = {int(k): v for k, v in data.items()}
            print("[OK] Model Statis & Label Berhasil Dimuat Ulang.")
        except Exception as e:
            print(f"[ERROR] Gagal memuat ulang Model Statis: {e}")
            
    # Reload dynamic model
    if os.path.exists(DYNAMIC_MODEL_PATH) and os.path.exists(LABEL_DYNAMIC_PATH):
        try:
            dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH)
            with open(LABEL_DYNAMIC_PATH, "r") as f:
                data = json.load(f)
                labels_dynamic = {int(k): v for k, v in data.items()}
            print("[OK] Model Dinamis & Label Berhasil Dimuat Ulang.")
        except Exception as e:
            print(f"[ERROR] Gagal memuat ulang Model Dinamis: {e}")

# =====================================================
# DATASET ACCESS HELPERS
# =====================================================
def load_dataset(mode: str):
    os.makedirs(STATIC_DATA_DIR, exist_ok=True)
    os.makedirs(DYNAMIC_DATA_DIR, exist_ok=True)
    
    if mode == "STATIC":
        X_path = os.path.join(STATIC_DATA_DIR, 'X_static.npy')
        y_path = os.path.join(STATIC_DATA_DIR, 'y_static.npy')
        if os.path.exists(X_path) and os.path.exists(y_path):
            try:
                X = np.load(X_path, allow_pickle=True)
                y = np.load(y_path, allow_pickle=True)
                return X, y
            except Exception as e:
                print(f"Error loading static npy files: {e}")
        return np.empty((0, 42, 3), dtype=np.float32), np.array([], dtype=str)
    else: # DYNAMIC
        X_path = os.path.join(DYNAMIC_DATA_DIR, 'X_dynamic.npy')
        y_path = os.path.join(DYNAMIC_DATA_DIR, 'y_dynamic.npy')
        if os.path.exists(X_path) and os.path.exists(y_path):
            try:
                X = np.load(X_path, allow_pickle=True)
                y = np.load(y_path, allow_pickle=True)
                return X, y
            except Exception as e:
                print(f"Error loading dynamic npy files: {e}")
        return np.empty((0, 30, 42, 3), dtype=np.float32), np.array([], dtype=str)

def save_dataset(mode: str, X: np.ndarray, y: np.ndarray):
    if mode == "STATIC":
        X_path = os.path.join(STATIC_DATA_DIR, 'X_static.npy')
        y_path = os.path.join(STATIC_DATA_DIR, 'y_static.npy')
    else:
        X_path = os.path.join(DYNAMIC_DATA_DIR, 'X_dynamic.npy')
        y_path = os.path.join(DYNAMIC_DATA_DIR, 'y_dynamic.npy')
        
    np.save(X_path, X)
    np.save(y_path, y)

# =====================================================
# PYDANTIC SCHEMAS
# =====================================================
class PredictionRequest(BaseModel):
    sequence: List[Any]  # Menerima koordinat landmarks dari Frontend
    mode: str            # 'STATIC' atau 'DYNAMIC'

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    status: str

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

class ModelInfo(BaseModel):
    static_model_loaded: bool
    dynamic_model_loaded: bool
    static_labels: int
    dynamic_labels: int

class CollectRequest(BaseModel):
    mode: str
    label: str
    landmarks: List[Any]

# =====================================================
# CORE PREDICTION LOGIC
# =====================================================
def normalize_frame_landmarks(flat_frame):
    """
    Normalisasi 1 frame landmark mentah dari frontend (126 nilai flat)
    menjadi format yang sama dengan extract_two_hands() di preprocessing.py.
    Output: numpy array shape (42, 3) yang sudah dinormalisasi.
    """
    arr = np.array(flat_frame, dtype=np.float32).reshape(42, 3)
    # Tangan pertama (index 0-20), tangan kedua (index 21-41)
    hand1 = arr[0:21]
    hand2 = arr[21:42]
    
    result = np.zeros((42, 3), dtype=np.float32)
    
    # Normalisasi tiap tangan jika ada landmark (bukan semua nol)
    if np.any(hand1 != 0):
        result[0:21] = normalize_landmarks(hand1)
    if np.any(hand2 != 0):
        result[21:42] = normalize_landmarks(hand2)
    
    return result

def run_inference(model, input_data):
    """Menjalankan fungsi predict TensorFlow di dalam thread terpisah."""
    return model.predict(input_data, verbose=0)

async def process_prediction(sequence_data: List[Any], mode: str) -> Dict[str, Any]:
    mode = mode.upper()
    
    if mode == "STATIC":
        if static_model is None:
            return {"prediction": "Model Statis tidak aktif", "confidence": 0.0, "status": "error"}
        
        # Normalisasi landmark mentah dari frontend (sama seperti extract_two_hands + normalize)
        normalized = normalize_frame_landmarks(sequence_data[-1])
        input_array = np.expand_dims(normalized, axis=0)  # shape (1, 42, 3)
        
        loop = asyncio.get_running_loop()
        res_probs = await loop.run_in_executor(executor, run_inference, static_model, input_array)
        
        pred_idx = int(np.argmax(res_probs[0]))
        confidence = float(np.max(res_probs[0]))
        raw_label = labels_static.get(pred_idx, "Tidak Diketahui")
        print(f"[STATIC] Raw: {raw_label} | Confidence: {confidence:.4f}")
        
        # Masukkan hasil ke penstabil getaran (smoother)
        smooth_label = smoother.process(raw_label, confidence)
        return {"prediction": smooth_label, "confidence": confidence * 100, "status": "success"}

    elif mode == "DYNAMIC":
        if dynamic_model is None:
            return {"prediction": "Model Dinamis tidak aktif", "confidence": 0.0, "status": "error"}
        
        # Saring baris sequence agar pas dengan ukuran input model CNN-LSTM (30 frame)
        if len(sequence_data) < SEQUENCE_LENGTH:
            return {"prediction": "Menunggu gerakan...", "confidence": 0.0, "status": "processing"}
            
        # Normalisasi setiap frame dalam sequence
        normalized_seq = [normalize_frame_landmarks(frame) for frame in sequence_data[-SEQUENCE_LENGTH:]]
        input_array = np.expand_dims(np.array(normalized_seq), axis=0)  # shape (1, 30, 42, 3)
        
        loop = asyncio.get_running_loop()
        res_probs = await loop.run_in_executor(executor, run_inference, dynamic_model, input_array)
        
        pred_idx = int(np.argmax(res_probs[0]))
        confidence = float(np.max(res_probs[0]))
        raw_label = labels_dynamic.get(pred_idx, "Tidak Diketahui")
        print(f"[DYNAMIC] Raw: {raw_label} | Confidence: {confidence:.4f}")
        
        smooth_label = smoother.process(raw_label, confidence)
        return {"prediction": smooth_label, "confidence": confidence * 100, "status": "success"}
        
    else:
        return {"prediction": "Mode Tidak Valid", "confidence": 0.0, "status": "error"}

# =====================================================
# API ENDPOINTS
# =====================================================
@app.get("/health")
def health_check():
    return {"status": "online"}

@app.get("/info", response_model=ModelInfo)
def get_model_info():
    return ModelInfo(
        static_model_loaded=static_model is not None,
        dynamic_model_loaded=dynamic_model is not None,
        static_labels=len(labels_static),
        dynamic_labels=len(labels_dynamic),
    )

@app.get("/labels")
def get_labels(mode: Optional[str] = Query("STATIC", description="Pilih mode: STATIC atau DYNAMIC")):
    mode = mode.upper()
    if mode == "STATIC":
        return {"mode": "STATIC", "labels": labels_static}
    if mode == "DYNAMIC":
        return {"mode": "DYNAMIC", "labels": labels_dynamic}
    raise HTTPException(status_code=400, detail="mode harus STATIC atau DYNAMIC")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        return await process_prediction(request.sequence, request.mode)
    except Exception as e:
        print(f"Error pada sistem API: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    responses = [
        await process_prediction(item.sequence, item.mode)
        for item in request.requests
    ]
    return {"predictions": responses}

@app.post("/smoother/clear")
def clear_smoother():
    smoother.clear_buffer()
    return {"status": "ok", "message": "Smoother buffer cleared"}

# =====================================================
# DATASET COLLECTION ENDPOINTS
# =====================================================
@app.get("/dataset/stats")
def get_dataset_stats():
    _, y_static = load_dataset("STATIC")
    _, y_dynamic = load_dataset("DYNAMIC")
    
    from collections import Counter
    static_counter = Counter(y_static)
    dynamic_counter = Counter(y_dynamic)
    
    return {
        "static": {
            "total_samples": len(y_static),
            "labels_count": dict(static_counter)
        },
        "dynamic": {
            "total_samples": len(y_dynamic),
            "labels_count": dict(dynamic_counter)
        }
    }

@app.post("/dataset/collect")
def collect_sample(request: CollectRequest):
    mode = request.mode.upper()
    label = request.label.strip()
    
    if not label:
        raise HTTPException(status_code=400, detail="Label tidak boleh kosong")
        
    if mode == "STATIC":
        if len(request.landmarks) != 126:
            # Handle if nested frame list is sent
            if len(request.landmarks) > 0 and isinstance(request.landmarks[0], list) and len(request.landmarks[0]) == 126:
                flat_frame = request.landmarks[0]
            else:
                raise HTTPException(status_code=400, detail=f"Landmark statis harus berisi 126 koordinat, didapat {len(request.landmarks)}")
        else:
            flat_frame = request.landmarks
            
        normalized = normalize_frame_landmarks(flat_frame)  # shape (42, 3)
        
        # Load & Append
        X, y = load_dataset("STATIC")
        X = np.concatenate([X, np.expand_dims(normalized, axis=0)], axis=0)
        y = np.append(y, label)
        
        # Save
        save_dataset("STATIC", X, y)
        return {"status": "success", "message": f"Landmark statis untuk label '{label}' disimpan.", "total_samples": len(y)}
        
    elif mode == "DYNAMIC":
        if len(request.landmarks) != 30:
            raise HTTPException(status_code=400, detail=f"Landmark dinamis harus berisi 30 frame, didapat {len(request.landmarks)}")
            
        normalized_sequence = []
        for i, frame in enumerate(request.landmarks):
            if not isinstance(frame, list) or len(frame) != 126:
                raise HTTPException(status_code=400, detail=f"Frame ke-{i} dari landmark dinamis harus berisi 126 koordinat")
            normalized_frame = normalize_frame_landmarks(frame)  # shape (42, 3)
            normalized_sequence.append(normalized_frame)
            
        normalized_sequence = np.array(normalized_sequence, dtype=np.float32)  # shape (30, 42, 3)
        
        # Load & Append
        X, y = load_dataset("DYNAMIC")
        X = np.concatenate([X, np.expand_dims(normalized_sequence, axis=0)], axis=0)
        y = np.append(y, label)
        
        # Save
        save_dataset("DYNAMIC", X, y)
        return {"status": "success", "message": f"Gerakan dinamis untuk label '{label}' disimpan.", "total_samples": len(y)}
        
    else:
        raise HTTPException(status_code=400, detail="Mode harus STATIC atau DYNAMIC")

@app.post("/dataset/train")
async def start_training(mode: Optional[str] = Query("STATIC", description="STATIC or DYNAMIC")):
    global training_task
    mode = mode.upper()
    if mode not in ["STATIC", "DYNAMIC"]:
        raise HTTPException(status_code=400, detail="mode harus STATIC atau DYNAMIC")
        
    if training_task.status == "training":
        raise HTTPException(status_code=400, detail=f"Pelatihan sedang berjalan untuk mode {training_task.mode}")
        
    # Jalankan training di background
    asyncio.create_task(run_training_subprocess(mode))
    return {"status": "success", "message": f"Memulai pelatihan model {mode} di latar belakang"}

@app.get("/dataset/train/status")
def get_training_status():
    global training_task
    return {
        "status": training_task.status,
        "mode": training_task.mode,
        "logs": training_task.logs[-50:]  # Berikan 50 log terakhir
    }

if __name__ == "__main__":
    import uvicorn
    
    # Tambahkan print log penanda server siap
    print("\n[START] Menjalankan Server VisiSign API...")
    print("[INFO] Akses dokumentasi API di: http://127.0.0.1:8001/docs")
    
    # Eksekusi server uvicorn pada port 8001
    uvicorn.run(app, host="127.0.0.1", port=8001)