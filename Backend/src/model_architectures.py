from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Dropout
from src.config import SEQUENCE_LENGTH, LANDMARK_POINTS, COORD_DIMS

def build_static_model(num_classes):
    """
    Arsitektur Multi-Layer Perceptron (MLP) dengan Batch Normalization dan Dropout
    untuk klasifikasi huruf dan angka (statis).
    """
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        Flatten(input_shape=(LANDMARK_POINTS, COORD_DIMS)),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def build_dynamic_model(num_classes):
    """
    Arsitektur CNN-LSTM untuk urutan video dinamis (Kata).
    Input shape: (T, 21, 3) 
    di mana T adalah SEQUENCE_LENGTH (contoh: 30 frame)
    """
    model = Sequential([
        # CNN (TimeDistributed) - Mengekstrak fitur spasial dari 21 titik x 3 koordinat pada setiap frame
        TimeDistributed(
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'), 
            input_shape=(SEQUENCE_LENGTH, LANDMARK_POINTS, COORD_DIMS)
        ),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        
        # LSTM - Mengekstrak pola temporal (waktu/pergerakan antar frame)
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Dense - Klasifikasi Akhir
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model