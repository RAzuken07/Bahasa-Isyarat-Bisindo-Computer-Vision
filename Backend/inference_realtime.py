import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque

from src.config import MODEL_DIR, SEQUENCE_LENGTH
from src.preprocessing import normalize_landmarks, extract_two_hands
from src.post_processing import PredictionSmoother

def load_labels(filename):
    with open(os.path.join(MODEL_DIR, filename), 'r') as f:
        # Konversi key string kembali ke int
        return {int(k): v for k, v in json.load(f).items()}

def main():
    # Load Models & Labels
    print("Memuat model dan label...")
    try:
        static_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'static_model.h5'))
        dynamic_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'dynamic_model.h5'))
        labels_static = load_labels('label_static.json')
        labels_dynamic = load_labels('label_dynamic.json')
    except Exception as e:
        print(f"Error memuat model/label: {e}. Pastikan Anda sudah menjalankan train_static.py dan train_dynamic.py!")
        return

    # Inisialisasi MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    # Inisialisasi State
    mode = "STATIC" # STATIC atau DYNAMIC
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    smoother = PredictionSmoother()
    current_prediction = "Menunggu..."

    cap = cv2.VideoCapture(0)
    print("\n=== KONTROL WEBCAM ===")
    print("Tekan '1' untuk Mode Huruf/Angka (Statis)")
    print("Tekan '2' untuk Mode Kata (Dinamis)")
    print("Tekan 'q' untuk Keluar")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Mirror
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Gambar visual landmark untuk semua tangan yang terdeteksi
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Ekstrak koordinat kedua tangan menjadi shape (42, 3)
            combined_landmarks = extract_two_hands(results.multi_hand_landmarks)

            if mode == "STATIC":
                input_data = np.expand_dims(combined_landmarks, axis=0) # shape (1, 42, 3)
                res = static_model.predict(input_data, verbose=0)[0]
                idx = np.argmax(res)
                current_prediction = smoother.process(labels_static[idx], res[idx])

            elif mode == "DYNAMIC":
                sequence_buffer.append(combined_landmarks)
                
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(np.array(sequence_buffer), axis=0) # shape (1, 30, 42, 3)
                    res = dynamic_model.predict(input_data, verbose=0)[0]
                    idx = np.argmax(res)
                    current_prediction = smoother.process(labels_dynamic[idx], res[idx])
        else:
            # Jika tidak ada tangan, reset buffer agar prediksi tidak "nyangkut"
            if mode == "DYNAMIC":
                sequence_buffer.clear()
            current_prediction = "Tidak ada tangan"
            smoother.clear_buffer()

        # UI: Tampilkan hasil di layar
        cv2.putText(frame, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Prediksi: {current_prediction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gesture Recognition', frame)

        # Kontrol Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            mode = "STATIC"
            smoother.clear_buffer()
            sequence_buffer.clear()
        elif key == ord('2'):
            mode = "DYNAMIC"
            smoother.clear_buffer()
            sequence_buffer.clear()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()