import os
import cv2
import numpy as np
import tensorflow as tf

IMAGE_DIR = "data/dataset/images/"
IMG_SIZE = (128, 128)

# Load and preprocess images
def load_images():
    images, labels = [], []
    for idx, img_name in enumerate(os.listdir(IMAGE_DIR)):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize
            images.append(img)
            labels.append(idx)  # Using index as label (change later for real labels)
    return np.array(images), np.array(labels)

X, y = load_images()
print(f"Loaded {len(X)} images.")
