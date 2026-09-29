"""
Input validation for Quantum Radiology Triage System.

This module prevents obvious normal photos, blank images,
very low-quality images, and invalid files from reaching the
Chest X-ray quantum classification model.

Important:
This is a prototype-level validation gate.
It does not replace a real DICOM/PACS validation workflow.
"""

import io
import numpy as np

from PIL import Image, UnidentifiedImageError


# ============================================================
# Validation limits
# ============================================================

MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64

# Colour photos usually have significant difference between R/G/B channels.
MAX_AVERAGE_CHANNEL_DIFFERENCE = 12.0

# Blank / almost constant images have very low variation.
MIN_GRAYSCALE_STD = 12.0

# Extremely narrow/wide images are unlikely to be normal chest X-rays.
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 1.80

# Prevent unusually huge files from being processed in demo.
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


# ============================================================
# Utility functions
# ============================================================

def calculate_colour_difference(rgb_array):
    """
    Returns average difference between R, G and B channels.

    For grayscale images:
    red ≈ green ≈ blue
    so difference is close to zero.

    For ordinary colour photos:
    channel differences are usually larger.
    """

    red = rgb_array[:, :, 0].astype(np.float32)
    green = rgb_array[:, :, 1].astype(np.float32)
    blue = rgb_array[:, :, 2].astype(np.float32)

    red_green_difference = np.mean(np.abs(red - green))
    green_blue_difference = np.mean(np.abs(green - blue))
    red_blue_difference = np.mean(np.abs(red - blue))

    average_difference = (
        red_green_difference
        + green_blue_difference
        + red_blue_difference
    ) / 3.0

    return float(average_difference)


def validate_chest_xray_image(image_bytes):
    """
    Validates uploaded image before model inference.

    Returns:
        tuple:
            is_valid (bool)
            message (str)
            image_info (dict)
    """

    image_info = {
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "channel_difference": None,
        "grayscale_std": None
    }

    # --------------------------------------------------------
    # 1. Validate file size
    # --------------------------------------------------------

    if not image_bytes:
        return (
            False,
            "Uploaded image is empty.",
            image_info
        )

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        return (
            False,
            "Image is too large. Please upload an image smaller than 10 MB.",
            image_info
        )

    # --------------------------------------------------------
    # 2. Check if file is a readable image
    # --------------------------------------------------------

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        # Reopen because verify() invalidates image object.
        image = Image.open(io.BytesIO(image_bytes))

    except UnidentifiedImageError:
        return (
            False,
            "Invalid file. Please upload a valid PNG, JPG, or JPEG chest X-ray image.",
            image_info
        )

    except Exception:
        return (
            False,
            "Unable to read the uploaded image. Please upload a valid chest X-ray image.",
            image_info
        )

    # --------------------------------------------------------
    # 3. Image dimension validation
    # --------------------------------------------------------

    width, height = image.size
    image_info["width"] = width
    image_info["height"] = height

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return (
            False,
            "Image resolution is too low. Please upload a clear chest X-ray image.",
            image_info
        )

    aspect_ratio = width / height
    image_info["aspect_ratio"] = round(aspect_ratio, 3)

    if (
        aspect_ratio < MIN_ASPECT_RATIO
        or aspect_ratio > MAX_ASPECT_RATIO
    ):
        return (
            False,
            "Image shape does not resemble a chest X-ray. "
            "Please upload a chest X-ray image only.",
            image_info
        )

    # --------------------------------------------------------
    # 4. Colour-photo rejection
    # --------------------------------------------------------

    rgb_image = image.convert("RGB")
    rgb_array = np.asarray(rgb_image)

    channel_difference = calculate_colour_difference(rgb_array)
    image_info["channel_difference"] = round(channel_difference, 3)

    if channel_difference > MAX_AVERAGE_CHANNEL_DIFFERENCE:
        return (
            False,
            "Rejected: This appears to be a normal colour photograph, "
            "not a grayscale chest X-ray.",
            image_info
        )

    # --------------------------------------------------------
    # 5. Blank / low-contrast image rejection
    # --------------------------------------------------------

    grayscale_image = image.convert("L")
    grayscale_array = np.asarray(
        grayscale_image,
        dtype=np.float32
    )

    grayscale_std = float(np.std(grayscale_array))
    image_info["grayscale_std"] = round(grayscale_std, 3)

    if grayscale_std < MIN_GRAYSCALE_STD:
        return (
            False,
            "Rejected: Image has too little contrast or appears blank. "
            "Please upload a clear chest X-ray.",
            image_info
        )

    # --------------------------------------------------------
    # Image accepted
    # --------------------------------------------------------

    return (
        True,
        "Valid grayscale X-ray-style image accepted.",
        image_info
    )