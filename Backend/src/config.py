import os

# ==========================================
# KONFIGURASI DIMENSI & MEDIA
# ==========================================
SEQUENCE_LENGTH = 30       # Jumlah frame (T) untuk Sliding Window video
LANDMARK_POINTS = 42       # Jumlah titik per tangan dari MediaPipe
COORD_DIMS = 3             # x, y, z

# ==========================================
# KONFIGURASI POST-PROCESSING (WEBCAM)
# ==========================================
PREDICTION_THRESHOLD = 0.5 # Confidence minimal (50%) agar prediksi dianggap valid
BUFFER_SIZE = 5            # Jumlah prediksi frame terakhir yang akan di-smoothing

# ==========================================
# KONFIGURASI PATH (DIREKTORI)
# ==========================================
# Secara otomatis mencari folder utama proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder Data
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
STATIC_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'static_landmarks')
DYNAMIC_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'dynamic_landmarks')

# Folder Model
MODEL_DIR = os.path.join(BASE_DIR, 'models')
STATIC_MODEL_PATH = os.path.join(MODEL_DIR, 'static_model.h5')
DYNAMIC_MODEL_PATH = os.path.join(MODEL_DIR, 'dynamic_model.h5')