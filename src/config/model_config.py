import os
from pathlib import Path

# Корневая директория проекта
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Пути к данным
DATA_DIR = ROOT_DIR / "multimodal-data"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATION_FILE = DATA_DIR / "annotation" / "annotation.csv"

# Пути к модели
WEIGHTS_DIR = ROOT_DIR / "weights"
BASE_WEIGHTS_PATH = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"
CONFIG_FILE = ROOT_DIR / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"

# Параметры обучения по умолчанию
DEFAULT_TRAIN_CONFIG = {
    'learning_rate': 1e-5,
    'batch_size': 1,
    'save_every': 10,
}

# Параметры детекции по умолчанию
DEFAULT_DETECTION_CONFIG = {
    'box_threshold': 0.25,
    'text_threshold': 0.2,
    'iou_threshold': 0.5,
}