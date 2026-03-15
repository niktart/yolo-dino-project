#!/usr/bin/env python
"""
Скрипт для аугментации датасета
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.data.augmenter import DatasetAugmenter


def main():
    print("=" * 50)
    print("🔄 АУГМЕНТАЦИЯ ДАТАСЕТА")
    print("=" * 50)
    
    # Проверяем существование датасета
    if not os.path.exists(settings.COMBINED_DATASET_PATH):
        print(f"❌ Датасет не найден: {settings.COMBINED_DATASET_PATH}")
        print("Сначала запустите скрипты для создания датасета")
        return
    
    # Создаем и запускаем аугментатор
    augmenter = DatasetAugmenter(
        dataset_path=settings.COMBINED_DATASET_PATH,
        min_images_per_class=settings.AUGMENTATION_MIN_IMAGES
    )
    
    stats = augmenter.run_pipeline()
    
    print("\n✅ Аугментация завершена успешно!")


if __name__ == "__main__":
    main()