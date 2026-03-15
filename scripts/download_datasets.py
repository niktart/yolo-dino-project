#!/usr/bin/env python
"""
Скрипт для скачивания и распаковки датасетов
Никакого копирования в combined_clean_bbox_уууу здесь не происходит!
"""

import sys
import os
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.utils.file_utils import extract_zip, extract_all_zips_from_dir


def find_random_zip():
    """Ищет zip файл в папке random"""
    random_dir = settings.DRIVE_RANDOM_PATH
    zip_files = glob.glob(os.path.join(random_dir, "*.zip"))
    
    if not zip_files:
        print(f"❌ В папке {random_dir} не найдено zip файлов!")
        return None
    
    print(f"📦 Найдены zip файлы в random: {[os.path.basename(f) for f in zip_files]}")
    # Ищем YOLO формат
    yolo_zips = [f for f in zip_files if 'yolov8' in f.lower()]
    if yolo_zips:
        return yolo_zips[0]
    return zip_files[0]


def main():
    print("=" * 50)
    print("📥 ЗАГРУЗКА ДАТАСЕТОВ")
    print("=" * 50)
    print("⚠️  ВНИМАНИЕ: Скрипт только распаковывает датасеты, не объединяет!")
    print("=" * 50)
    
    # Создаем базовую папку для данных
    os.makedirs("/content/data", exist_ok=True)
    
    # 1. Распаковываем random датасет
    print("\n📦 RANDOM ДАТАСЕТ")
    print("-" * 30)
    
    random_zip = find_random_zip()
    if random_zip:
        print(f"📦 Распаковка random датасета в {settings.RANDOM_DATASET_PATH}...")
        extract_zip(random_zip, settings.RANDOM_DATASET_PATH)
        print(f"✅ Random датасет распакован в {settings.RANDOM_DATASET_PATH}")
    else:
        print("❌ Random датасет не найден")
    
    # 2. Распаковываем датасеты из combined папки
    print("\n📦 COMBINED ДАТАСЕТЫ")
    print("-" * 30)
    
    if os.path.exists(settings.DRIVE_COMBINED_PATH):
        # Распаковываем каждый датасет в отдельную папку в /content/data
        extract_all_zips_from_dir(settings.DRIVE_COMBINED_PATH, "/content/data")
    else:
        print(f"❌ Папка не найдена: {settings.DRIVE_COMBINED_PATH}")
    
    # 3. Показываем итоговую структуру
    print("\n📁 РАСПАКОВАННЫЕ ДАТАСЕТЫ:")
    print("=" * 50)
    
    if os.path.exists("/content/data"):
        for item in sorted(os.listdir("/content/data")):
            item_path = os.path.join("/content/data", item)
            if os.path.isdir(item_path):
                yaml_path = os.path.join(item_path, "data.yaml")
                if os.path.exists(yaml_path):
                    print(f"  ✅ {item}/")
                else:
                    print(f"  ⚠️  {item}/ (нет data.yaml)")
    
    print(f"\n📁 Random датасет: {settings.RANDOM_DATASET_PATH}")
    if os.path.exists(settings.RANDOM_DATASET_PATH):
        print(f"  ✅ Распакован")
    
    print("\n✅ Загрузка завершена! Датасеты готовы к анализу.")


if __name__ == "__main__":
    main()