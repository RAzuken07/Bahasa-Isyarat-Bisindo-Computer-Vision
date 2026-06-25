import os
import cv2
import numpy as np
import random
import glob

# Configurasi
RAW_IMG_DIR = os.path.join('data', 'raw', 'images')
TARGET_COUNT = 350

def augment_image(image):
    """
    Melakukan augmentasi gambar secara acak (rotasi, zoom, translasi).
    """
    rows, cols, _ = image.shape
    
    # 1. Rotasi acak (-15 hingga 15 derajat)
    angle = random.uniform(-15, 15)
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    
    # 2. Scaling (Zoom in/out acak 0.9 hingga 1.1)
    scale = random.uniform(0.9, 1.1)
    M_rot[0, 0] *= scale
    M_rot[0, 1] *= scale
    M_rot[1, 0] *= scale
    M_rot[1, 1] *= scale
    
    img_aug = cv2.warpAffine(image, M_rot, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    
    # 3. Brightness/Contrast sedikit
    alpha = random.uniform(0.8, 1.2) # Kontras
    beta = random.uniform(-10, 10)   # Kecerahan
    img_aug = cv2.convertScaleAbs(img_aug, alpha=alpha, beta=beta)
    
    return img_aug

def balance_dataset():
    if not os.path.exists(RAW_IMG_DIR):
        print(f"Directory {RAW_IMG_DIR} not found!")
        return

    classes = os.listdir(RAW_IMG_DIR)
    
    for class_name in classes:
        class_path = os.path.join(RAW_IMG_DIR, class_name)
        if not os.path.isdir(class_path): continue
            
        images = glob.glob(os.path.join(class_path, '*.jpg')) + glob.glob(os.path.join(class_path, '*.png'))
        count = len(images)
        
        if count < TARGET_COUNT:
            print(f"Kelas '{class_name}': {count} gambar. Melakukan augmentasi...")
            diff = TARGET_COUNT - count
            
            for i in range(diff):
                # Pilih gambar acak dari yang sudah ada
                img_path = random.choice(images)
                img = cv2.imread(img_path)
                
                if img is None: continue
                    
                # Augmentasi
                aug_img = augment_image(img)
                
                # Simpan
                new_name = f"aug_{i}_{os.path.basename(img_path)}"
                new_path = os.path.join(class_path, new_name)
                cv2.imwrite(new_path, aug_img)
                
            print(f" -> Selesai augmentasi kelas '{class_name}'. Total sekarang: {TARGET_COUNT}")
        else:
            print(f"Kelas '{class_name}': {count} gambar (Sudah cukup)")

if __name__ == "__main__":
    balance_dataset()
