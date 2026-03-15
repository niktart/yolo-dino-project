#!/usr/bin/env python
"""
Скрипт для конвертации датасета из YOLO формата в COCO
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.data.yolo_to_coco_converter import YOLOToCOCOConverter


def parse_args():
    parser = argparse.ArgumentParser(description='Конвертация YOLO -> COCO')
    parser.add_argument('--dataset_root', type=str, default=settings.COMBINED_DATASET_PATH,
                        help='Корневая папка датасета')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Сплиты для конвертации')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("🔄 КОНВЕРТАЦИЯ YOLO -> COCO")
    print("=" * 50)
    
    # Проверяем существование датасета
    if not os.path.exists(args.dataset_root):
        print(f"❌ Датасет не найден: {args.dataset_root}")
        return
    
    # Создаем конвертер
    converter = YOLOToCOCOConverter(
        dataset_root=args.dataset_root,
        splits=args.splits
    )
    
    # Конвертируем
    results = converter.convert_all()
    
    print(f"\n✅ Конвертация завершена!")
    for split, path in results.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()