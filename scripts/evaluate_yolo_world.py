#!/usr/bin/env python
"""
Скрипт для оценки обученной модели YOLO-World
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.models.yolo_world import YOLOWorldModel
from src.evaluation.metrics import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(description='Оценка YOLO-World')
    parser.add_argument('--weights', type=str, required=True,
                        help='Путь к весам модели')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Сплит для оценки')
    parser.add_argument('--batch', type=int, default=16,
                        help='Размер батча')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("📊 ОЦЕНКА МОДЕЛИ YOLO-World")
    print("=" * 50)
    
    # Проверяем наличие датасета
    data_yaml = os.path.join(settings.COMBINED_DATASET_PATH, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"❌ Датасет не найден: {data_yaml}")
        return
    
    # Проверяем наличие весов
    if not os.path.exists(args.weights):
        print(f"❌ Веса не найдены: {args.weights}")
        return
    
    # Создаем модель
    model = YOLOWorldModel(model_path=args.weights)
    
    # Валидация
    val_result = model.validate(
        data_yaml=data_yaml,
        split=args.split,
        batch=args.batch,
        weights_path=args.weights
    )
    
    print(f"\n✅ Оценка завершена!")
    print(f"📁 Метрики сохранены: {val_result['metrics_path']}")


if __name__ == "__main__":
    main()