"""Central project settings.

The scripts are intended to be launched from the project root, but all paths
are resolved from this file so they also work when called from another folder.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "vehicle_type_cnn.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
TRAINING_CONFIG_PATH = MODELS_DIR / "training_config.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

APP_DIR = PROJECT_ROOT / "app"
UPLOAD_DIR = APP_DIR / "static" / "uploads"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LAYERS = 40
SEED = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
}
