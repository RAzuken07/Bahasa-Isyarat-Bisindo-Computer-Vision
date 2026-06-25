import os
import sys

# ==========================================
# PERBAIKAN PATH (PENTING)
# ==========================================
# Memberitahu Python letak folder utama (Backend) agar bisa membaca modul 'src'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import mediapipe as mp
import numpy as np

# Import dari folder src
from src.preprocessing import normalize_landmarks, extract_two_hands
from src.config import SEQUENCE_LENGTH

# ==========================================
# INISIALISASI MEDIAPIPE (MAKSIMAL 2 TANGAN)
# ==========================================
mp_hands = mp.solutions.hands
hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_dynamic = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# ==========================================
# KONFIGURASI DIREKTORI (ABSOLUTE PATH)
# ==========================================
RAW_IMG_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'images')
RAW_VID_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'videos')
OUT_STATIC_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'static_landmarks')
OUT_DYNAMIC_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'dynamic_landmarks')

def extract_from_images():
    print("Mengekstrak landmark statis (Gambar) untuk 2 tangan...")
    os.makedirs(OUT_STATIC_DIR, exist_ok=True)
    
    X_static, y_static = [], []
    
    if not os.path.exists(RAW_IMG_DIR):
        print(f"Folder {RAW_IMG_DIR} tidak ditemukan!")
        return

    # Loop setiap subfolder (nama kelas)
    for class_name in os.listdir(RAW_IMG_DIR):
        class_path = os.path.join(RAW_IMG_DIR, class_name)
        if not os.path.isdir(class_path): continue
            
        # INDIKATOR PROSES
        print(f"  -> Memproses gambar huruf/angka: {class_name} ...")
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands_static.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Menggunakan fungsi 2 tangan (menghasilkan shape 42, 3)
                combined_landmarks = extract_two_hands(results.multi_hand_landmarks)
                X_static.append(combined_landmarks)
                y_static.append(class_name)
    
    # Simpan sebagai .npy
    if X_static:
        np.save(os.path.join(OUT_STATIC_DIR, 'X_static.npy'), np.array(X_static))
        np.save(os.path.join(OUT_STATIC_DIR, 'y_static.npy'), np.array(y_static))
        print(f"Selesai! Tersimpan {len(X_static)} sampel statis.\n")

def extract_from_videos():
    print("Mengekstrak landmark dinamis (Video) untuk 2 tangan...")
    os.makedirs(OUT_DYNAMIC_DIR, exist_ok=True)
    
    X_dynamic, y_dynamic = [], []
    
    if not os.path.exists(RAW_VID_DIR):
        print(f"Folder {RAW_VID_DIR} tidak ditemukan!")
        return

    # Loop setiap subfolder (nama kelas)
    for class_name in os.listdir(RAW_VID_DIR):
        class_path = os.path.join(RAW_VID_DIR, class_name)
        if not os.path.isdir(class_path): continue
            
        # INDIKATOR PROSES
        print(f"  -> Memproses video kata: {class_name} ...")
            
        for vid_name in os.listdir(class_path):
            vid_path = os.path.join(class_path, vid_name)
            cap = cv2.VideoCapture(vid_path)
            
            sequence_data = []
            
            while cap.isOpened() and len(sequence_data) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_dynamic.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    # Menggunakan fungsi 2 tangan (menghasilkan shape 42, 3)
                    combined_landmarks = extract_two_hands(results.multi_hand_landmarks)
                    sequence_data.append(combined_landmarks)
                else:
                    # Jika tidak ada tangan, isi dengan array nol berukuran (42, 3)
                    sequence_data.append(np.zeros((42, 3)))
            
            cap.release()
            
            # Handling jika jumlah frame video kurang dari SEQUENCE_LENGTH
            while len(sequence_data) < SEQUENCE_LENGTH:
                sequence_data.append(np.zeros((42, 3)))
                
            X_dynamic.append(sequence_data)
            y_dynamic.append(class_name)
            
    # Simpan sebagai .npy
    if X_dynamic:
        np.save(os.path.join(OUT_DYNAMIC_DIR, 'X_dynamic.npy'), np.array(X_dynamic))
        np.save(os.path.join(OUT_DYNAMIC_DIR, 'y_dynamic.npy'), np.array(y_dynamic))
        print(f"Selesai! Tersimpan {len(X_dynamic)} sampel dinamis dengan sequence {SEQUENCE_LENGTH}.")

if __name__ == "__main__":
    extract_from_images()
    extract_from_videos()