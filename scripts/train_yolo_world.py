#!/usr/bin/env python
"""
Скрипт для обучения модели YOLO-World
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.models.yolo_world import YOLOWorldModel
from src.utils.drive_utils import DriveSaver


def parse_args():
    parser = argparse.ArgumentParser(description='Обучение YOLO-World')
    parser.add_argument('--epochs', type=int, default=settings.YOLO_WORLD_CONFIG['default_epochs'],
                        help='Количество эпох')
    parser.add_argument('--batch', type=int, default=settings.YOLO_WORLD_CONFIG['default_batch'],
                        help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=settings.YOLO_WORLD_CONFIG['default_imgsz'],
                        help='Размер изображения')
    parser.add_argument('--lr0', type=float, default=settings.YOLO_WORLD_CONFIG['default_lr0'],
                        help='Начальный learning rate')
    parser.add_argument('--freeze', type=int, default=settings.YOLO_WORLD_CONFIG['default_freeze'],
                        help='Количество замороженных слоев')
    parser.add_argument('--mixup', type=float, default=settings.YOLO_WORLD_CONFIG['default_mixup'],
                        help='Коэффициент mixup')
    parser.add_argument('--model', type=str, default=settings.YOLO_WORLD_CONFIG['model_name'],
                        help='Название модели')
    parser.add_argument('--save_to_drive', action='store_true',
                        help='Сохранять результаты в Google Drive')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("🤖 ОБУЧЕНИЕ YOLO-World")
    print("=" * 50)
    
    # Проверяем наличие датасета
    data_yaml = os.path.join(settings.COMBINED_DATASET_PATH, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"❌ Датасет не найден: {data_yaml}")
        print("Сначала запустите скрипты для создания датасета")
        return
    
    # Создаем модель
    model = YOLOWorldModel(model_path=args.model)
    
    # Обучаем
    result = model.train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        freeze=args.freeze,
        mixup=args.mixup
    )
    
    print(f"\n✅ Обучение завершено!")
    print(f"📁 Веса сохранены: {result['best_weights_path']}")
    print(f"⏱️ Время обучения: {result['training_time_formatted']}")
    
    # Валидация
    print("\n🔍 Запуск валидации...")
    val_result = model.validate(
        data_yaml=data_yaml,
        split="val",
        weights_path=result["best_weights_path"]
    )
    
    # Сохранение в Google Drive (опционально)
    if args.save_to_drive:
        print("\n💾 Сохранение в Google Drive...")
        drive_saver = DriveSaver()
        drive_saver.mount_drive()
        drive_saver.save_experiment_results(
            weights_path=result['best_weights_path'],
            metrics_path=val_result['metrics_path'],
            experiment_name=result['experiment_name']
        )


if __name__ == "__main__":
    main()