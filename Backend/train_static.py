"""
Train Model MLP
"""


import os
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt  # Tambahan untuk plot
import seaborn as sns            # Tambahan untuk visualisasi matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report # Tambahan metrik


# Import dari folder src
from src.config import STATIC_DATA_DIR, MODEL_DIR, STATIC_MODEL_PATH
from src.model_architectures import build_static_model

def augment_landmarks(X, y, num_augments=4):
    """
    Mengaugmentasi data landmark dengan rotasi acak pada sumbu Z,
    penskalaan acak, dan penambahan noise Gaussian kecil.
    Hanya mengaugmentasi koordinat tangan yang aktif (bukan padding nol).
    """
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # Masukkan sampel asli
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # Lakukan augmentasi sebanyak num_augments kali
        for _ in range(num_augments):
            augmented_sample = np.zeros_like(X[i])
            
            # Proses masing-masing dari 2 tangan
            for hand_idx in range(2):
                start_idx = hand_idx * 21
                end_idx = start_idx + 21
                hand = X[i, start_idx:end_idx]
                
                # Hanya jika tangan terdeteksi (tidak bernilai nol semua)
                if np.any(hand != 0):
                    # A. Rotasi Z acak (-15 sampai +15 derajat)
                    angle = np.random.uniform(-15, 15)
                    theta = np.radians(angle)
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([
                        [c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]
                    ])
                    aug_hand = np.dot(hand, R.T)
                    
                    # B. Scaling acak (0.9 sampai 1.1)
                    scale = np.random.uniform(0.9, 1.1)
                    aug_hand = aug_hand * scale
                    
                    # C. Jittering (noise Gaussian kecil)
                    noise = np.random.normal(0, 0.01, size=aug_hand.shape)
                    aug_hand = aug_hand + noise
                    
                    augmented_sample[start_idx:end_idx] = aug_hand
                else:
                    augmented_sample[start_idx:end_idx] = hand
            
            X_augmented.append(augmented_sample)
            y_augmented.append(y[i])
            
    return np.array(X_augmented), np.array(y_augmented)

def train():
    print("Memuat data statis...")
    X_path = os.path.join(STATIC_DATA_DIR, 'X_static.npy')
    y_path = os.path.join(STATIC_DATA_DIR, 'y_static.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Data tidak ditemukan di {STATIC_DATA_DIR}. Jalankan extract_features.py dulu!")
        return
 
    X = np.load(X_path)
    y = np.load(y_path)
 
    # Encode label teks ('A', 'B') menjadi angka (0, 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
 
    # Simpan mapping label ke JSON agar bisa dibaca saat inference webcam
    os.makedirs(MODEL_DIR, exist_ok=True)
    label_mapping = {int(index): str(label) for index, label in enumerate(le.classes_)}
    with open(os.path.join(MODEL_DIR, 'label_static.json'), 'w') as f:
        json.dump(label_mapping, f)
 
    # Split data training dan testing (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
 


    # Lakukan augmentasi geometris hanya pada data training yang kini sudah seimbang
    print("Melakukan augmentasi data latih...")
    X_train, y_train = augment_landmarks(X_train, y_train, num_augments=4)

    print(f"Jumlah kelas: {num_classes}")
    print(f"Data latih (setelah SMOTE + Augmentasi): {X_train.shape}, Data uji: {X_test.shape}")
 
    # Bangun dan latih model
    model = build_static_model(num_classes)
    
    # Callback untuk berhenti jika akurasi sudah bagus
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
 
    print("\nMulai pelatihan model statis...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

    # ==========================================
    # EVALUASI: Akurasi, Classification Report & Confusion Matrix
    # ==========================================
    print("\n--- Evaluasi Model Statis ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss pada data uji: {loss:.4f}")
    print(f"Akurasi model pada data uji: {accuracy * 100:.2f}%\n")

    # 1. Lakukan Prediksi pada data uji
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1) # Mengubah probabilitas menjadi indeks kelas

    # 2. Cetak Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=np.arange(num_classes), target_names=le.classes_, zero_division=0))

    # 3. Hitung dan Visualisasikan Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    
    plt.figure(figsize=(10, 8)) # Diperbesar karena kelas alfabet/angka cukup banyak
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Model Statis')
    plt.ylabel('Label Sebenarnya (True)')
    plt.xlabel('Label Prediksi (Predicted)')
    
    # Simpan gambar agar tidak hilang
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix_static.png')
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"\nVisualisasi Confusion Matrix disimpan di: {cm_path}")
    
    # Simpan model
    model.save(STATIC_MODEL_PATH)
    print(f"\nModel statis berhasil disimpan di: {STATIC_MODEL_PATH}")

if __name__ == "__main__":
    train()