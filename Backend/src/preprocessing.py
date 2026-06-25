import numpy as np

def normalize_landmarks(landmarks):
    """
    Menormalkan koordinat landmark relatif terhadap pergelangan tangan (landmark 0)
    dan menormalkan skala berdasarkan jarak pergelangan tangan ke pangkal jari tengah.
    Input: list of 21 landmark points [[x, y, z], [x, y, z], ...]
    Output: numpy array shape (21, 3)
    """
    landmarks = np.array(landmarks)
    # 1. Translasi (Wrist ke 0,0,0)
    translated_landmarks = landmarks - landmarks[0]
    
    # 2. Normalisasi Skala (Scale Invariant)
    # Menggunakan jarak antara wrist (0) dan MCP jari tengah (9) sebagai ukuran telapak tangan
    scale = np.linalg.norm(translated_landmarks[9])
    
    if scale < 1e-5:
        # Fallback ke jarak terjauh jika koordinat index 9 tidak terdeteksi/bermasalah
        scale = np.max(np.linalg.norm(translated_landmarks, axis=1))
        
    if scale > 1e-5:
        normalized_landmarks = translated_landmarks / scale
    else:
        normalized_landmarks = translated_landmarks
        
    return normalized_landmarks

# Buka src/preprocessing.py, tambahkan kode ini di bagian paling bawah:

def extract_two_hands(multi_hand_landmarks):
    """
    Mengambil hingga 2 tangan dan mengembalikannya dalam array berukuran (42, 3).
    Jika hanya 1 tangan, separuh array akan berisi angka nol.
    """
    # Siapkan array kosong untuk 42 titik x 3 koordinat
    combined_landmarks = np.zeros((42, 3))
    
    if multi_hand_landmarks:
        for i, hand_landmarks in enumerate(multi_hand_landmarks):
            if i >= 2: break # Batasi maksimal 2 tangan saja
            
            # Ambil koordinat mentah
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            
            # Normalisasi
            norm_landmarks = normalize_landmarks(landmarks)
            
            # Masukkan ke array: 
            # Index 0-20 untuk tangan pertama, Index 21-41 untuk tangan kedua
            start_idx = i * 21
            end_idx = start_idx + 21
            combined_landmarks[start_idx:end_idx] = norm_landmarks
            
    return combined_landmarks