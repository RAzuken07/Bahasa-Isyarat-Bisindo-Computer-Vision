import os
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt  # Tambahan untuk plot
import seaborn as sns            # Tambahan untuk visualisasi matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report # Tambahan metrik
from imblearn.over_sampling import SMOTE
from collections import Counter

# Import dari folder src
from src.config import DYNAMIC_DATA_DIR, MODEL_DIR, DYNAMIC_MODEL_PATH
from src.model_architectures import build_dynamic_model

def train():
    print("Memuat data dinamis...")
    X_path = os.path.join(DYNAMIC_DATA_DIR, 'X_dynamic.npy')
    y_path = os.path.join(DYNAMIC_DATA_DIR, 'y_dynamic.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Data tidak ditemukan di {DYNAMIC_DATA_DIR}. Jalankan extract_features.py dulu!")
        return

    X = np.load(X_path)
    y = np.load(y_path)

    # Encode label teks ('Halo', 'Tolong') menjadi angka (0, 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Simpan mapping label ke JSON
    os.makedirs(MODEL_DIR, exist_ok=True)
    label_mapping = {int(index): str(label) for index, label in enumerate(le.classes_)}
    with open(os.path.join(MODEL_DIR, 'label_dynamic.json'), 'w') as f:
        json.dump(label_mapping, f)

    # Split data training dan testing (Data uji tetap murni sebelum SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # ====================================================================
    # INTEGRASI SMOTE: DATA DINAMIS (ND -> 2D -> SMOTE -> ND)
    # ====================================================================
    print(f"\n[SMOTE] Distribusi kelas data latih sebelum SMOTE: {Counter(y_train)}")
    
    # Safety Check: Duplikasi kelas dengan sampel < 2 agar SMOTE tidak crash
    for c in np.unique(y_train):
        c_count = np.sum(y_train == c)
        if c_count < 2:
            idx = np.where(y_train == c)[0]
            X_train = np.concatenate([X_train, X_train[idx]], axis=0)
            y_train = np.concatenate([y_train, y_train[idx]], axis=0)
            print(f"[SMOTE Fix] Duplikasi kelas {c} karena hanya memiliki {c_count} sampel di y_train.")
            
    # 1. Simpan bentuk (shape) asli data X_train secara dinamis
    bentuk_asli_X = X_train.shape
    n_samples = bentuk_asli_X[0]
    
    # 2. Flatten semua dimensi setelah sampel menjadi 2D agar bisa diproses SMOTE
    X_train_flatten = X_train.reshape(n_samples, -1)
    
    # 3. Tentukan k_neighbors secara dinamis untuk mengantisipasi kelas minoritas ekstrem
    min_sampel_kelas = min(Counter(y_train).values())
    k_tetangga = 5 if min_sampel_kelas > 5 else max(1, min_sampel_kelas - 1)
    
    # 4. Jalankan SMOTE pada data 2D
    print(f"[SMOTE] Menyeimbangkan data dinamis dengan k_neighbors={k_tetangga}...")
    smote = SMOTE(k_neighbors=k_tetangga, random_state=42)
    X_train_resampled_flatten, y_train_resampled = smote.fit_resample(X_train_flatten, y_train)
    
    # 5. Kembalikan bentuk data latih hasil SMOTE ke dimensi aslinya
    X_train = X_train_resampled_flatten.reshape((-1,) + bentuk_asli_X[1:])
    y_train = y_train_resampled
    
    print(f"[SMOTE] Distribusi kelas data latih setelah SMOTE: {Counter(y_train)}\n")
    # ====================================================================

    print(f"Jumlah kelas: {num_classes}")
    print(f"Data latih (setelah SMOTE): {X_train.shape}, Data uji: {X_test.shape}")

    # Bangun dan latih model
    model = build_dynamic_model(num_classes)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\nMulai pelatihan model dinamis...")
    model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])

    # ==========================================
    # EVALUASI: Akurasi, Classification Report & Confusion Matrix
    # ==========================================
    print("\n--- Evaluasi Model ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss pada data uji: {loss:.4f}")
    print(f"Akurasi model pada data uji: {accuracy * 100:.2f}%\n")

    # 1. Lakukan Prediksi pada data uji
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1) # Mengubah probabilitas menjadi indeks kelas

    # 2. Cetak Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 3. Hitung dan Visualisasikan Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Model Dinamis')
    plt.ylabel('Label Sebenarnya (True)')
    plt.xlabel('Label Prediksi (Predicted)')
    
    # Simpan gambar agar tidak hilang
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"\nVisualisasi Confusion Matrix disimpan di: {cm_path}")
    
    # Simpan model
    model.save(DYNAMIC_MODEL_PATH)
    print(f"\nModel dinamis berhasil disimpan di: {DYNAMIC_MODEL_PATH}")

if __name__ == "__main__":
    train()