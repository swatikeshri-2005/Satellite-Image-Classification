from pathlib import Path  # noqa: I001


# ============================================
# PROJECT DIRECTORIES
# ============================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = BASE_DIR / "data" / "raw" / "EuroSAT"

# Directory for trained models
MODEL_DIR = BASE_DIR / "models"

# Directory for generated outputs
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================
# MODEL
# ============================================

# Saved model
MODEL_PATH = MODEL_DIR / "satellite_classifier.keras"


# ============================================
# IMAGE SETTINGS
# ============================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32


# ============================================
# TRAINING SETTINGS
# ============================================

EPOCHS = 15

LEARNING_RATE = 0.0001


# ============================================
# DATASET SETTINGS
# ============================================

VALIDATION_SPLIT = 0.2

SEED = 42