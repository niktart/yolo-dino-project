from datetime import datetime
import torch

import os
from pathlib import Path

# Пути к Google Drive
DRIVE_BASE_PATH = "/content/drive/MyDrive/term_work"
DRIVE_RANDOM_PATH = os.path.join(DRIVE_BASE_PATH, "random")
DRIVE_COMBINED_PATH = os.path.join(DRIVE_BASE_PATH, "combined_dataset")

# Локальные пути
LOCAL_BASE_PATH = "/content"
DATA_ROOT = os.path.join(LOCAL_BASE_PATH, "data")

# Путь к исходным датасетам (для анализа)
SOURCE_DATASETS_PATH = DATA_ROOT  # /content/data - здесь лежат все исходные датасеты

# Путь к итоговому объединенному датасету
COMBINED_DATASET_PATH = os.path.join(DATA_ROOT, "combined_clean_bbox_уууу")  # будет создан позже

# Путь к random датасету
RANDOM_DATASET_PATH = os.path.join(LOCAL_BASE_PATH, "random_dataset")

# Параметры сплитов
SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}
RANDOM_STATE = 42

# Маппинг классов
RENAME_MAP = {
    "apple": "apple_unknown",
    "Apple": "apple_unknown",
    "Carrot": "carrot",
    "carrote": "carrot",
    "sprite": "sprite_bottle",
    "Sprite": "sprite_can",
    "fanta": "fanta_bottle",
    "Fanta": "fanta_can",
    "coca-cola": "cola_bottle",
    "Cola": "cola_can",
    "Onion": "onion",
    "Potato": "potato",
    "tomate": "tomato",
    "pomme de terre": "potato",
    "bellpepper": "bell pepper",
}

# Параметры для добавления random датасета
RANDOM_CLASS_START_ID = 71
RANDOM_CLASS_END_ID = 76

# Параметры для аугментации
AUGMENTATION_MIN_IMAGES = 150


# Параметры для YOLO-World
YOLO_WORLD_CONFIG = {
    "model_name": "yolov8s-worldv2.pt",
    "default_epochs": 3,
    "default_imgsz": 640,
    "default_batch": 16,
    "default_lr0": 0.001,
    "default_lrf": 0.01,
    "default_freeze": -5,
    "default_mixup": 0.1,
    "workers": 0,
    "cache": False
}



# Параметры для сохранения экспериментов
EXPERIMENT_NAME = "to_compare_with_dino_{epochs}e_b{batch}_f{freeze}_m{mixup}"

# Параметры для detection rate
DETECTION_RATE_CONFIG = {
    "default_conf_threshold": 0.15,
    "default_iou_threshold": 0.5,
    "confidence_thresholds": [0.1, 0.15, 0.2, 0.25, 0.3]
}



import os
import torch
from pathlib import Path

# ==================== ПУТИ ====================
# Google Drive
DRIVE_BASE_PATH = "/content/drive/MyDrive/term_work"
DRIVE_WEIGHTS_PATH = os.path.join(DRIVE_BASE_PATH, "weights")

# Локальные пути
HOME_PATH = "/content"
PROJECT_PATH = os.path.join(HOME_PATH, "Grounding-Dino-FineTuning-main")  # ВАЖНО!
MULTIMODAL_PATH = os.path.join(PROJECT_PATH, "multimodal-data")
WEIGHTS_PATH = os.path.join(PROJECT_PATH, "weights")

# Данные
DATA_PATH = "/content/data"
SOURCE_DATASETS_PATH = DATA_PATH
COMBINED_DATASET_PATH = os.path.join(DATA_PATH, "combined_clean_bbox_уууу")

# ==================== МОДЕЛЬ ====================
GROUNDING_DINO_CONFIG = {
    "config_path": os.path.join(PROJECT_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    "pretrained_weights": os.path.join(WEIGHTS_PATH, "groundingdino_swint_ogc.pth"),
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==================== ОБУЧЕНИЕ ====================
TRAINING_CONFIG = {
    "default_epochs": 3,
    "default_lr": 5e-6,
    "save_epoch": 1,
    "save_path": os.path.join(WEIGHTS_PATH, "model_")
}