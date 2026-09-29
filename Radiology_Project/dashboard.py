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
# File names
# ============================================================

QUANTUM_MODEL_FILE = "quantum_model.pkl"
PCA_MODEL_FILE = "pca_model.pkl"
SCALED_TRAIN_FILE = "quantum_train_features_scaled.npy"
CLASSICAL_MODEL_FILE = "classical_cnn.pth"


# ============================================================
# CNN architecture
# Must match the training architecture exactly
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
# Load models once per Streamlit server
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
        filename
        for filename in required_files
        if not os.path.exists(filename)
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required model file(s): "
            + ", ".join(missing_files)
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
    Convert a validated uploaded image to training input shape:

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
# Prediction
# ============================================================

def predict_chest_xray(image_bytes, model_data):
    """
    Validate input and run:
    Image -> CNN -> PCA -> scaling -> quantum kernel -> QSVM.
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
        reduced_features
    )

    new_quantum_kernel = (
        model_data["quantum_kernel"].evaluate(
            x_vec=scaled_features,
            y_vec=model_data["quantum_train_features"]
        )
    )

    prediction = int(
        model_data["quantum_classifier"].predict(
            new_quantum_kernel
        )[0]
    )

    probabilities = (
        model_data["quantum_classifier"].predict_proba(
            new_quantum_kernel
        )[0]
    )

    confidence = float(
        np.max(probabilities)
    )

    normal_probability = float(
        probabilities[0]
    )

    pneumonia_probability = float(
        probabilities[1]
    )

    # Original dataset labels:
    # 0 = Normal
    # 1 = Pneumonia
    if prediction == 1:
        diagnosis = "Pneumonia"
        priority = "urgent"
    else:
        diagnosis = "Normal"
        priority = "non-urgent"

    return {
        "success": True,
        "is_valid_xray": True,
        "validation_message": validation_message,
        "image_info": image_info,
        "prediction": prediction,
        "diagnosis": diagnosis,
        "priority": priority,
        "confidence": confidence,
        "normal_probability": normal_probability,
        "pneumonia_probability": pneumonia_probability
    }


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("About This Prototype")

st.sidebar.markdown("""
- **Use case:** Chest X-ray triage
- **Classes:** Normal vs Pneumonia
- **Classical layer:** CNN feature extraction
- **Quantum layer:** Quantum kernel + SVM classification
- **Output:** Urgent / Non-urgent priority
""")

st.sidebar.warning("""
**Research prototype only**

This system is for hackathon demonstration and educational use.
It is not a clinical diagnostic tool and must not be used for
patient-care decisions.
""")


# ============================================================
# Load trained files
# ============================================================

try:
    with st.spinner("Loading trained CNN and quantum model..."):
        model_data = load_models()

    st.sidebar.success("✅ Model loaded successfully")

except Exception as error:
    st.sidebar.error("❌ Model loading failed")
    st.error(
        "Unable to load the trained model files: "
        f"{str(error)}"
    )
    st.stop()


# ============================================================
# Main title
# ============================================================

st.title("🏥 Quantum Chest X-ray Triage System")

st.markdown(
    "**Hybrid Classical-Quantum Model for Chest X-ray "
    "Normal vs Pneumonia Triage**"
)


# ============================================================
# Upload section
# ============================================================

st.header("Upload Chest X-ray")

st.info("""
Upload only a **grayscale chest X-ray image** in PNG, JPG, or JPEG format.

The input-validation layer rejects:
- Normal colour photographs
- Blank or very low-contrast images
- Very low-resolution images
- Invalid image files
""")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# Preview and prediction
# ============================================================

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        try:
            uploaded_file.seek(0)

            preview_image = Image.open(
                uploaded_file
            )

            st.image(
                preview_image,
                caption="Uploaded Image",
                use_container_width=True
            )

        except Exception:
            st.error(
                "Unable to preview this image. "
                "Upload a valid PNG, JPG, or JPEG file."
            )

    with col2:
        st.subheader("Input Status")

        st.write(
            f"**File name:** {uploaded_file.name}"
        )

        st.write(
            f"**File type:** {uploaded_file.type}"
        )

        st.write(
            f"**File size:** "
            f"{uploaded_file.size / 1024:.1f} KB"
        )

    if st.button(
        "🔮 Validate and Predict with Quantum AI",
        type="primary"
    ):
        with st.spinner(
            "Validating image and running quantum prediction..."
        ):
            try:
                uploaded_file.seek(0)

                image_bytes = uploaded_file.getvalue()

                result = predict_chest_xray(
                    image_bytes,
                    model_data
                )

                if not result["success"]:
                    st.error("❌ Image Rejected")

                    st.warning(
                        result.get(
                            "error",
                            "This image cannot be accepted for "
                            "Chest X-ray analysis."
                        )
                    )

                    image_info = result.get(
                        "image_info"
                    )

                    if image_info:
                        with st.expander(
                            "Validation details"
                        ):
                            st.json(image_info)

                    st.stop()

                st.success(
                    result.get(
                        "validation_message",
                        "Valid chest X-ray-style image accepted."
                    )
                )

                st.divider()
                st.subheader("AI Triage Result")

                diagnosis = result["diagnosis"]
                priority = result["priority"]
                confidence = result["confidence"]

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    if priority == "urgent":
                        st.error(
                            f"🚨 **URGENT:** {diagnosis}"
                        )

                        st.warning(
                            "This image is classified as a "
                            "pneumonia-pattern proxy. "
                            "Prioritize radiologist review."
                        )

                    else:
                        st.success(
                            f"✓ **NON-URGENT:** {diagnosis}"
                        )

                        st.info(
                            "This image is classified as "
                            "normal by the prototype model."
                        )

                with result_col2:
                    st.metric(
                        "Model Confidence",
                        f"{confidence * 100:.1f}%"
                    )

                    st.metric(
                        "Normal Probability",
                        f"{result['normal_probability'] * 100:.1f}%"
                    )

                    st.metric(
                        "Pneumonia Probability",
                        f"{result['pneumonia_probability'] * 100:.1f}%"
                    )

                image_info = result.get(
                    "image_info"
                )

                if image_info:
                    with st.expander(
                        "Input Validation Details"
                    ):
                        st.write(
                            f"**Image dimensions:** "
                            f"{image_info.get('width')} × "
                            f"{image_info.get('height')}"
                        )

                        st.write(
                            f"**Aspect ratio:** "
                            f"{image_info.get('aspect_ratio')}"
                        )

                        st.write(
                            f"**Colour-channel difference:** "
                            f"{image_info.get('channel_difference')}"
                        )

                        st.write(
                            f"**Grayscale contrast score:** "
                            f"{image_info.get('grayscale_std')}"
                        )

            except Exception as error:
                st.error(
                    f"Prediction failed: {str(error)}"
                )


# ============================================================
# Demo worklist
# ============================================================

st.divider()
st.header("📋 Radiologist Worklist — Demo View")

st.caption(
    "This is sample visual data for demonstrating priority-based "
    "radiology workflow. It is not connected to patient records."
)

worklist_data = pd.DataFrame({
    "Case ID": [
        "Demo-001",
        "Demo-002",
        "Demo-003",
        "Demo-004",
        "Demo-005"
    ],
    "Priority": [
        "🚨 Urgent",
        "✓ Non-Urgent",
        "🚨 Urgent",
        "✓ Non-Urgent",
        "🚨 Urgent"
    ],
    "Queue Time": [
        "5 min",
        "45 min",
        "10 min",
        "60 min",
        "8 min"
    ],
    "Status": [
        "Awaiting review",
        "Routine queue",
        "Awaiting review",
        "Routine queue",
        "Awaiting review"
    ]
})

st.dataframe(
    worklist_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# Pipeline explanation
# ============================================================

st.divider()
st.header("⚛️ Hybrid Quantum Pipeline")

pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = (
    st.columns(4)
)

with pipeline_col1:
    st.info(
        "**1. Validation**\n\n"
        "Reject non-X-ray-style images."
    )

with pipeline_col2:
    st.info(
        "**2. Classical CNN**\n\n"
        "Extract image features."
    )

with pipeline_col3:
    st.info(
        "**3. PCA + Encoding**\n\n"
        "Compress to four quantum inputs."
    )

with pipeline_col4:
    st.info(
        "**4. Quantum Kernel**\n\n"
        "Classify Normal vs Pneumonia."
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.markdown(
    "**Quantum Innovation:** Hybrid CNN + PCA + quantum-kernel "
    "classification for chest X-ray triage.  \n"
    "**Safety Layer:** Input validation rejects obvious "
    "non-radiology colour photos before prediction."
)