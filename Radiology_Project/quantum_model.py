"""
Step 3: Train quantum model with medical feature map
"""

import os
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ============================================================
# Configuration
# ============================================================

TRAIN_FEATURES_FILE = "train_features.npy"
TEST_FEATURES_FILE = "test_features.npy"
TRAIN_LABELS_FILE = "train_labels.npy"
TEST_LABELS_FILE = "test_labels.npy"

MODEL_FILE = "quantum_model.pkl"
SCALED_TRAIN_FILE = "quantum_train_features_scaled.npy"

# Increase only after this version works
MAX_TRAIN_SAMPLES = 200
MAX_TEST_SAMPLES = 80

RANDOM_SEED = 42
NUMBER_OF_FEATURES = 4


# ============================================================
# Utility functions
# ============================================================

def load_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"File not found: {filename}\n"
            "Keep all .npy files in the same folder as quantum_model.py."
        )

    return np.load(filename)


def choose_balanced_samples(X, y, total_samples, seed):
    """
    Select approximately equal samples from each class.
    This prevents the quantum subset from containing
    mostly only one class.
    """

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    if len(classes) < 2:
        raise ValueError("The labels must contain at least two classes.")

    samples_per_class = total_samples // len(classes)
    selected_indices = []

    for class_value in classes:
        class_indices = np.where(y == class_value)[0]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {class_value}. "
                f"Available: {len(class_indices)}, "
                f"required: {samples_per_class}."
            )

        chosen = rng.choice(
            class_indices,
            size=samples_per_class,
            replace=False
        )

        selected_indices.extend(chosen.tolist())

    rng.shuffle(selected_indices)
    selected_indices = np.array(selected_indices)

    return X[selected_indices], y[selected_indices]


# ============================================================
# Load original data
# ============================================================

print("Loading feature files...")

X_train = load_file(TRAIN_FEATURES_FILE).astype(float)
X_test = load_file(TEST_FEATURES_FILE).astype(float)
y_train = load_file(TRAIN_LABELS_FILE).astype(int)
y_test = load_file(TEST_LABELS_FILE).astype(int)

print("Original training shape:", X_train.shape)
print("Original testing shape :", X_test.shape)
print("Original training labels:", np.bincount(y_train))
print("Original testing labels :", np.bincount(y_test))


# ============================================================
# Reduce data for quantum kernel calculation
# ============================================================

print("\nSelecting balanced quantum subsets...")

X_train, y_train = choose_balanced_samples(
    X_train,
    y_train,
    MAX_TRAIN_SAMPLES,
    RANDOM_SEED
)

X_test, y_test = choose_balanced_samples(
    X_test,
    y_test,
    MAX_TEST_SAMPLES,
    RANDOM_SEED + 1
)

print("Selected training shape:", X_train.shape)
print("Selected testing shape :", X_test.shape)
print("Selected training labels:", np.bincount(y_train))
print("Selected testing labels :", np.bincount(y_test))


# ============================================================
# Select four features
# ============================================================

if X_train.ndim != 2 or X_test.ndim != 2:
    raise ValueError(
        "Feature arrays must have shape: (number_of_samples, number_of_features)."
    )

if X_train.shape[1] < NUMBER_OF_FEATURES:
    raise ValueError(
        f"At least {NUMBER_OF_FEATURES} features are required."
    )

X_train = X_train[:, :NUMBER_OF_FEATURES]
X_test = X_test[:, :NUMBER_OF_FEATURES]

print("\nQuantum training feature shape:", X_train.shape)
print("Quantum testing feature shape :", X_test.shape)


# ============================================================
# Scale features
# ============================================================

print("\nScaling features...")

scaler = MinMaxScaler(feature_range=(-1, 1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# The backend needs this file to calculate
# the new sample's kernel against training samples.
np.save(SCALED_TRAIN_FILE, X_train)


# ============================================================
# Create the quantum feature map
# ============================================================

print("\nCreating quantum feature map...")

feature_map = zz_feature_map(
    feature_dimension=NUMBER_OF_FEATURES,
    reps=1,
    entanglement="linear"
)

print("\nQuantum feature map:")
print(feature_map.draw(output="text"))


# ============================================================
# Create the quantum kernel
# ============================================================

print("\nCreating fidelity quantum kernel...")

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)


# ============================================================
# Calculate quantum kernel matrices
# ============================================================

print("\nComputing training quantum kernel...")
print("This compares the selected training samples.")

K_train = quantum_kernel.evaluate(
    x_vec=X_train
)

print("Training quantum kernel completed.")

print("\nComputing testing quantum kernel...")
print("This compares test samples with training samples.")

K_test = quantum_kernel.evaluate(
    x_vec=X_test,
    y_vec=X_train
)

print("Testing quantum kernel completed.")


# ============================================================
# Train the SVM using the quantum kernel
# ============================================================

print("\nTraining SVM with quantum kernel...")

model = SVC(
    kernel="precomputed",
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_SEED
)

model.fit(K_train, y_train)

print("SVM training completed.")


# ============================================================
# Evaluate
# ============================================================

print("\nEvaluating quantum model...")

predictions = model.predict(K_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)

print("\n========================================")
print("Quantum Model Results")
print("========================================")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Pneumonia"],
        zero_division=0
    )
)


# ============================================================
# Save model
# ============================================================

model_data = {
    "quantum_kernel": quantum_kernel,
    "classifier": model,
    "scaler": scaler,
    "feature_map": feature_map,
    "feature_count": NUMBER_OF_FEATURES,
    "random_seed": RANDOM_SEED,
    "train_sample_count": len(X_train),
    "test_sample_count": len(X_test)
}

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model_data, file)

print("\n========================================")
print("Saved files")
print("========================================")
print(f"Model:              {MODEL_FILE}")
print(f"Scaled train data:  {SCALED_TRAIN_FILE}")
print("\nQuantum model completed successfully.")