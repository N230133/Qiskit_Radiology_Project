"""
Streamlit-only deployment version of Quantum Chest X-ray Triage.

This file contains:
- Input validation
- CNN architecture and feature extraction
- PCA transformation
- Quantum kernel + QSVM inference
- Streamlit dashboard interface

No Flask backend is needed for this deployment version.
"""

import io
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image

from image_validator import validate_chest_xray_image


# ============================================================
# Streamlit page setup
# Must be the first Streamlit command
# ============================================================

st.set_page_config(
    page_title="Quantum Chest X-ray Triage",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# Model file paths
#
# BASE_DIR is the folder containing this dashboard.py file.
# In your GitHub repository, it is:
# Qiskit_Radiology_Project/Radiology_Project/
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

QUANTUM_MODEL_FILE = BASE_DIR / "quantum_model.pkl"
PCA_MODEL_FILE = BASE_DIR / "pca_model.pkl"
SCALED_TRAIN_FILE = BASE_DIR / "quantum_train_features_scaled.npy"
CLASSICAL_MODEL_FILE = BASE_DIR / "classical_cnn.pth"


# ============================================================
# CNN architecture
# Must match classical_baseline.py exactly
# ============================================================

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return x


class ClassicalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = CNNBackbone()

        self.fc1 = nn.Linear(
            in_features=32 * 32 * 32,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=2
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# Load models once
# ============================================================

@st.cache_resource
def load_models():
    required_files = [
        QUANTUM_MODEL_FILE,
        PCA_MODEL_FILE,
        SCALED_TRAIN_FILE,
        CLASSICAL_MODEL_FILE
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        readable_missing_files = "\n".join(
            str(file_path)
            for file_path in missing_files
        )

        raise FileNotFoundError(
            "Missing required model file(s):\n"
            + readable_missing_files
        )

    with open(QUANTUM_MODEL_FILE, "rb") as file:
        quantum_model_data = pickle.load(file)

    with open(PCA_MODEL_FILE, "rb") as file:
        pca = pickle.load(file)

    quantum_kernel = quantum_model_data["quantum_kernel"]
    quantum_classifier = quantum_model_data["classifier"]
    quantum_scaler = quantum_model_data["scaler"]

    quantum_train_features = np.load(
        SCALED_TRAIN_FILE
    )

    checkpoint = torch.load(
        CLASSICAL_MODEL_FILE,
        map_location="cpu"
    )

    full_cnn_model = ClassicalCNN()

    full_cnn_model.load_state_dict(
        checkpoint
    )

    full_cnn_model.eval()

    backbone = full_cnn_model.backbone
    backbone.eval()

    return {
        "pca": pca,
        "quantum_kernel": quantum_kernel,
        "quantum_classifier": quantum_classifier,
        "quantum_scaler": quantum_scaler,
        "quantum_train_features": quantum_train_features,
        "backbone": backbone
    }


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(image_bytes):
    """
    Convert validated image to the exact training tensor format.

    Output shape:
    (1, 1, 128, 128)
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    image = image.resize(
        (128, 128)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    )

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# ============================================================
# CNN feature extraction
# ============================================================

def extract_cnn_features(backbone, image_tensor):
    with torch.no_grad():
        features = backbone(
            image_tensor
        ).numpy()

    return features


# ============================================================
# Prediction pipeline
# ============================================================

def predict_chest_xray(image_bytes, model_data):
    """
    Full inference flow:

    Image validation
        ↓
    CNN feature extraction
        ↓
    PCA reduction
        ↓
    Feature scaling
        ↓
    Quantum kernel calculation
        ↓
    QSVM prediction
    """

    is_valid, validation_message, image_info = (
        validate_chest_xray_image(image_bytes)
    )

    if not is_valid:
        return {
            "success": False,
            "is_valid_xray": False,
            "error": validation_message,
            "image_info": image_info
        }

    image_tensor = preprocess_image(
        image_bytes
    )

    cnn_features = extract_cnn_features(
        model_data["backbone"],
        image_tensor
    )

    reduced_features = model_data["pca"].transform(
        cnn_features
    )

    reduced_features = reduced_features[:, :4]

    scaled_features = model_data["quantum_scaler"].transform(
        red
