#!/usr/bin/env python
"""
Скрипт для добавления random датасета к основному датасету
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.data.random_dataset_adder import RandomDatasetAdder


def main():
    print("=" * 50)
    print("➕ ДОБАВЛЕНИЕ RANDOM ДАТАСЕТА")
    print("=" * 50)
    
    # Проверяем существование директорий
    if not os.path.exists(settings.RANDOM_DATASET_PATH):
        print(f"❌ Random датасет не найден: {settings.RANDOM_DATASET_PATH}")
        print("Сначала запустите: python scripts/download_datasets.py")
        return
    
    if not os.path.exists(settings.COMBINED_DATASET_PATH):
        print(f"❌ Основной датасет не найден: {settings.COMBINED_DATASET_PATH}")
        print("Сначала запустите: python scripts/merge_datasets.py")
        return
    
    # Создаем и запускаем пайплайн
    adder = RandomDatasetAdder(
        target_dataset_path=settings.COMBINED_DATASET_PATH,
        random_dataset_path=settings.RANDOM_DATASET_PATH
    )
    
    id_map = adder.run_pipeline(
        start_id=settings.RANDOM_CLASS_START_ID,
        end_id=settings.RANDOM_CLASS_END_ID
    )
    
    print("\n✅ Скрипт завершен успешно!")


if __name__ == "__main__":
    main()