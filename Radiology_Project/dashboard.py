"""
Step 5: Streamlit dashboard for Quantum Chest X-ray Triage demo.

The dashboard:
1. Accepts PNG/JPG/JPEG uploads
2. Sends image to Flask backend
3. Backend rejects obvious normal photos / invalid images
4. Displays prediction only for accepted X-ray-style inputs
"""


import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os


# ============================================================
# Page setup
# ============================================================


st.set_page_config(
    page_title="Quantum Chest X-ray Triage",
    page_icon="🏥",
    layout="wide"
)


DEFAULT_API_URL = "http://127.0.0.1:5000"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    DEFAULT_API_URL
).rstrip("/")

API_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"



# ============================================================
# Title
# ============================================================


st.title("🏥 Quantum Chest X-ray Triage System")

st.markdown(
    "**Hybrid Classical-Quantum Model for Chest X-ray "
    "Normal vs Pneumonia Triage**"
)



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
# API health check
# ============================================================


try:
    health_response = requests.get(
        HEALTH_URL,
        timeout=3
    )

    if health_response.status_code == 200:
        st.sidebar.success("✅ Backend API connected")
    else:
        st.sidebar.error("❌ Backend API is not responding")

except requests.exceptions.RequestException:
    st.sidebar.error("❌ Backend API is offline")



# ============================================================
# Input instructions
# ============================================================


st.header("Upload Chest X-ray")

st.info("""
Upload only a **grayscale chest X-ray image** in PNG, JPG, or JPEG format.

The backend validates the input before prediction and rejects:
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
# Upload preview and prediction
# ============================================================


if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        try:
            preview_image = Image.open(uploaded_file)

            st.image(
                preview_image,
                caption="Uploaded Image",
                use_container_width=True
            )

        except Exception:
            st.error(
                "Unable to preview this image. "
                "Please upload a valid PNG, JPG, or JPEG file."
            )

    with col2:
        st.subheader("Input Status")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File type:** {uploaded_file.type}")
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

                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=120
                )

                try:
                    result = response.json()
                except ValueError:
                    st.error(
                        "Backend returned an invalid response. "
                        "Please check backend_api.py."
                    )
                    st.stop()

                # ------------------------------------------------
                # Invalid image / backend validation failed
                # ------------------------------------------------

                if response.status_code != 200:
                    st.error("❌ Image Rejected")

                    st.warning(
                        result.get(
                            "error",
                            "This image cannot be accepted for "
                            "Chest X-ray analysis."
                        )
                    )

                    image_info = result.get("image_info")

                    if image_info:
                        with st.expander("Validation details"):
                            st.json(image_info)

                    st.stop()

                # ------------------------------------------------
                # General backend failure
                # ------------------------------------------------

                if not result.get("success", False):
                    st.error(
                        result.get(
                            "error",
                            "Prediction failed."
                        )
                    )
                    st.stop()

                # ------------------------------------------------
                # Valid X-ray and successful model prediction
                # ------------------------------------------------

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

                image_info = result.get("image_info")

                if image_info:
                    with st.expander("Input Validation Details"):
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

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to Flask backend."
                )

                st.info(
                    "Open a new terminal and run:\n\n"
                    "`python backend_api.py`"
                )

            except requests.exceptions.Timeout:
                st.error(
                    "Prediction timed out. "
                    "The quantum kernel calculation may be taking too long."
                )

            except Exception as error:
                st.error(
                    f"Unexpected error: {str(error)}"
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

pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = st.columns(4)

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
    "**Safety Layer:** Input validation rejects obvious non-radiology "
    "colour photos before prediction."
)