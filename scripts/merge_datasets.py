#!/usr/bin/env python
"""
Скрипт для объединения датасетов
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.data.merger import DatasetMerger


def main():
    print("=" * 50)
    print("🔄 ОБЪЕДИНЕНИЕ ДАТАСЕТОВ")
    print("=" * 50)
    
    # Проверяем наличие исходных датасетов
    if not os.path.exists(settings.SOURCE_DATASETS_PATH):
        print(f"❌ Исходные датасеты не найдены: {settings.SOURCE_DATASETS_PATH}")
        print("Сначала запустите: python scripts/download_datasets.py")
        return
    
    # Проверяем, не существует ли уже целевой датасет
    if os.path.exists(settings.COMBINED_DATASET_PATH):
        print(f"⚠️ Целевой датасет уже существует: {settings.COMBINED_DATASET_PATH}")
        response = input("Перезаписать? (y/n): ")
        if response.lower() != 'y':
            print("❌ Операция отменена")
            return
        import shutil
        shutil.rmtree(settings.COMBINED_DATASET_PATH)
        print("🗑️ Старый датасет удален")
    
    # Создаем мерджер
    merger = DatasetMerger(
        source_dir=settings.SOURCE_DATASETS_PATH,  # читаем из /content/data
        target_dir=settings.COMBINED_DATASET_PATH,  # создаем в /content/data/combined_clean_bbox_уууу
        rename_map=settings.RENAME_MAP,
        split_ratios=settings.SPLIT_RATIOS,
        random_state=settings.RANDOM_STATE
    )
    
    # Запускаем пайплайн
    try:
        df_stats = merger.run_pipeline()
        
        print("\n" + "=" * 50)
        print("✅ ОБЪЕДИНЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 50)
        print(f"📁 Итоговый датасет: {merger.target_dir}")
        print(f"📊 Всего классов: {len(merger.final_names)}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при объединении: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()