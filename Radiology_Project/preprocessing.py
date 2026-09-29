"""
Step 1: Download and preprocess Chest X-Ray dataset
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download dataset (manual step)
# Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Unzip to 'chest-xray-pneumonia/' folder

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
DATA_DIR = 'chest_xray'

# Load training data
print("Loading dataset...")
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Extract all images and labels
print("Extracting images and labels...")
X_train_list, y_train_list = [], []
for i in range(len(train_data)):
    batch_X, batch_y = train_data[i]
    X_train_list.append(batch_X)
    y_train_list.append(batch_y)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

X_test_list, y_test_list = [], []
for i in range(len(test_data)):
    batch_X, batch_y = test_data[i]
    X_test_list.append(batch_X)
    y_test_list.append(batch_y)

X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Save preprocessed data
print("Saving preprocessed data...")
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("✓ Data preprocessing complete!")