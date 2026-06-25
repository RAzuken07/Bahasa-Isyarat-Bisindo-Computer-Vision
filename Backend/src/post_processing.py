from collections import deque
from collections import Counter
from src.config import PREDICTION_THRESHOLD, BUFFER_SIZE

class PredictionSmoother:
    def __init__(self):
        # deque akan otomatis membuang data terlama jika panjangnya melebihi BUFFER_SIZE
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def process(self, predicted_class, confidence_score):
        """
        Menyaring prediksi berdasarkan threshold dan mengambil suara terbanyak (modus)
        dari beberapa frame terakhir.
        """
        # 1. Confidence Threshold
        if confidence_score >= PREDICTION_THRESHOLD:
            self.buffer.append(predicted_class)
        else:
            self.buffer.append("Tidak Jelas")

        # 2. Smoothing (Majority Voting)
        if len(self.buffer) == BUFFER_SIZE:
            # Hitung kemunculan tiap kelas dalam buffer
            counts = Counter(self.buffer)
            most_common_class, most_common_count = counts.most_common(1)[0]
            
            # Pastikan kelas terbanyak mendominasi lebih dari separuh isi buffer
            if most_common_class != "Tidak Jelas" and most_common_count > (BUFFER_SIZE / 2):
                return most_common_class
                
        return "Menunggu gerakan..."
        
    def clear_buffer(self):
        """Mereset buffer (berguna jika berganti mode Ejaan <-> Kata)"""
        self.buffer.clear()