#!/usr/bin/env python
"""
Скрипт для конвертации COCO JSON в CSV для Grounding DINO
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.data.coco_to_csv_converter import COKOToCSVConverter


def parse_args():
    parser = argparse.ArgumentParser(description='Конвертация COCO -> CSV')
    parser.add_argument('--dataset_root', type=str, default=settings.COMBINED_DATASET_PATH,
                        help='Корневая папка датасета')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                        help='Сплиты для конвертации')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("🔄 КОНВЕРТАЦИЯ COCO -> CSV")
    print("=" * 50)
    
    # Проверяем существование датасета
    if not os.path.exists(args.dataset_root):
        print(f"❌ Датасет не найден: {args.dataset_root}")
        return
    
    # Создаем конвертер
    converter = COKOToCSVConverter(dataset_root=args.dataset_root)
    
    # Конвертируем
    results = converter.convert_all_splits(splits=args.splits)
    
    print(f"\n✅ Конвертация завершена!")
    for split, path in results.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()