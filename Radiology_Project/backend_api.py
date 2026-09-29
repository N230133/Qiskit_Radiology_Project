"""
Step 4: Flask API for real-time prediction.

This API:
1. Receives image from Streamlit dashboard
2. Validates that it is chest-X-ray-style input
3. Rejects obvious normal colour photos / invalid images
4. Runs CNN -> PCA -> Quantum Kernel -> SVM prediction
"""


from flask import Flask, request, jsonify
from flask_cors import CORS


import io
import os
import pickle


import numpy as np
import torch
import torch.nn as nn


from PIL import Image


from image_validator import validate_chest_xray_image



# ============================================================
# Flask setup
# ============================================================


app = Flask(__name__)
CORS(app)



# ============================================================
# File names
# ============================================================


QUANTUM_MODEL_FILE = "quantum_model.pkl"
PCA_MODEL_FILE = "pca_model.pkl"
SCALED_TRAIN_FILE = "quantum_train_features_scaled.npy"
CLASSICAL_MODEL_FILE = "classical_cnn.pth"



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
# Check required files
# ============================================================


required_files = [
    QUANTUM_MODEL_FILE,
    PCA_MODEL_FILE,
    SCALED_TRAIN_FILE,
    CLASSICAL_MODEL_FILE
]


for filename in required_files:
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Required file is missing: {filename}"
        )



# ============================================================
# Load quantum model
# ============================================================


print("Loading quantum model...")


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


print("Quantum model loaded successfully.")



# ============================================================
# Load complete classical CNN
# ============================================================


print("Loading classical CNN...")


checkpoint = torch.load(
    CLASSICAL_MODEL_FILE,
    map_location="cpu"
)


full_cnn_model = ClassicalCNN()


full_cnn_model.load_state_dict(checkpoint)


full_cnn_model.eval()


backbone = full_cnn_model.backbone
backbone.eval()


print("Classical CNN loaded successfully.")



# ============================================================
# Image preprocessing
# ============================================================


def preprocess_image(image_bytes):
    """
    Convert validated image into the exact tensor format used in training.

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
# Feature extraction
# ============================================================


def extract_cnn_features(image_tensor):
    with torch.no_grad():
        features = backbone(
            image_tensor
        ).numpy()

    return features



# ============================================================
# Health endpoint
# ============================================================


@app.route(
    "/health",
    methods=["GET"]
)
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "message": "Quantum Chest X-ray Triage API is running"
    })



# ============================================================
# Input-validation endpoint
# Optional: useful for testing validation separately
# ============================================================


@app.route(
    "/validate-image",
    methods=["POST"]
)
def validate_image():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "is_valid_xray": False,
                "error": "No image file was uploaded."
            }), 400

        uploaded_file = request.files["image"]
        image_bytes = uploaded_file.read()

        is_valid, message, image_info = validate_chest_xray_image(
            image_bytes
        )

        return jsonify({
            "success": is_valid,
            "is_valid_xray": is_valid,
            "message": message,
            "image_info": image_info
        }), 200 if is_valid else 400

    except Exception as error:
        return jsonify({
            "success": False,
            "is_valid_xray": False,
            "error": str(error)
        }), 500



# ============================================================
# Prediction endpoint
# ============================================================


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    try:
        # ----------------------------------------------------
        # 1. File check
        # ----------------------------------------------------

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "is_valid_xray": False,
                "error": "No image file was uploaded."
            }), 400

        uploaded_file = request.files["image"]
        image_bytes = uploaded_file.read()

        if len(image_bytes) == 0:
            return jsonify({
                "success": False,
                "is_valid_xray": False,
                "error": "Uploaded image is empty."
            }), 400

        # ----------------------------------------------------
        # 2. Validate image before ML/quantum prediction
        # ----------------------------------------------------

        is_valid, validation_message, image_info = (
            validate_chest_xray_image(image_bytes)
        )

        if not is_valid:
            return jsonify({
                "success": False,
                "is_valid_xray": False,
                "error": validation_message,
                "image_info": image_info
            }), 400

        # ----------------------------------------------------
        # 3. Image -> tensor
        # ----------------------------------------------------

        image_tensor = preprocess_image(
            image_bytes
        )

        # ----------------------------------------------------
        # 4. Tensor -> CNN features
        # ----------------------------------------------------

        cnn_features = extract_cnn_features(
            image_tensor
        )

        # ----------------------------------------------------
        # 5. CNN features -> PCA features
        # ----------------------------------------------------

        reduced_features = pca.transform(
            cnn_features
        )

        reduced_features = reduced_features[
            :, :4
        ]

        # ----------------------------------------------------
        # 6. Scale exactly as training
        # ----------------------------------------------------

        scaled_features = quantum_scaler.transform(
            reduced_features
        )

        # ----------------------------------------------------
        # 7. Calculate quantum kernel against saved train data
        # ----------------------------------------------------

        new_quantum_kernel = quantum_kernel.evaluate(
            x_vec=scaled_features,
            y_vec=quantum_train_features
        )

        # ----------------------------------------------------
        # 8. Quantum-kernel SVM prediction
        # ----------------------------------------------------

        prediction = quantum_classifier.predict(
            new_quantum_kernel
        )[0]

        probabilities = quantum_classifier.predict_proba(
            new_quantum_kernel
        )[0]

        confidence = float(
            np.max(probabilities)
        )

        normal_probability = float(probabilities[0])
        pneumonia_probability = float(probabilities[1])

        # ----------------------------------------------------
        # 9. Convert output labels
        # ----------------------------------------------------

        # Dataset labels:
        # 0 = Normal
        # 1 = Pneumonia
        if int(prediction) == 1:
            diagnosis = "Pneumonia"
            priority = "urgent"
        else:
            diagnosis = "Normal"
            priority = "non-urgent"

        # ----------------------------------------------------
        # 10. Return result
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "is_valid_xray": True,
            "validation_message": validation_message,
            "image_info": image_info,
            "prediction": int(prediction),
            "diagnosis": diagnosis,
            "priority": priority,
            "confidence": confidence,
            "normal_probability": normal_probability,
            "pneumonia_probability": pneumonia_probability
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "is_valid_xray": False,
            "error": str(error)
        }), 500



# ============================================================
# Start server
# ============================================================


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )