#!/usr/bin/env python
"""
Скрипт для расчета доли полностью распознанных изображений для YOLO
"""

import sys
import os
import argparse
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.evaluation.yolo_detection_rate import YOLODetectionRateCalculator
from ultralytics import YOLOWorld


def parse_args():
    parser = argparse.ArgumentParser(description='Расчет detection rate для YOLO')
    parser.add_argument('--weights', type=str, required=True,
                        help='Путь к весам модели')
    parser.add_argument('--dataset_root', type=str, default=settings.COMBINED_DATASET_PATH,
                        help='Корневая папка датасета')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Сплит для оценки')
    parser.add_argument('--conf', type=float, default=0.15,
                        help='Порог уверенности')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='Порог IoU')
    parser.add_argument('--save_results', action='store_true',
                        help='Сохранять результаты в CSV')
    parser.add_argument('--analyze_errors', action='store_true',
                        help='Анализировать ошибки')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.1, 0.15, 0.2, 0.25, 0.3],
                        help='Пороги для анализа')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("🎯 РАСЧЕТ DETECTION RATE ДЛЯ YOLO")
    print("=" * 50)
    
    # Проверяем существование весов
    if not os.path.exists(args.weights):
        print(f"❌ Веса не найдены: {args.weights}")
        return
    
    # Проверяем существование датасета
    images_dir = os.path.join(args.dataset_root, args.split, "images")
    labels_dir = os.path.join(args.dataset_root, args.split, "labels")
    
    if not os.path.exists(images_dir):
        print(f"❌ Директория с изображениями не найдена: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"❌ Директория с аннотациями не найдена: {labels_dir}")
        return
    
    # Загружаем модель
    print(f"\n📥 Загрузка модели из {args.weights}...")
    model = YOLOWorld(args.weights)
    model.eval()
    print("✅ Модель загружена")
    
    # Создаем калькулятор
    calculator = YOLODetectionRateCalculator(model)
    
    # Рассчитываем detection rate
    print(f"\n📊 Расчет для сплита: {args.split}")
    detection_rate, df = calculator.calculate_detection_rate(
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_results=args.save_results,
        output_csv=f"detection_rate_{args.split}_conf{args.conf}_iou{args.iou}.csv"
    )
    
    # Анализируем ошибки (опционально)
    if args.analyze_errors and not df.empty:
        print("\n🔍 Анализ ошибок...")
        error_stats = calculator.analyze_errors(df)
        
        # Сохраняем статистику ошибок
        if args.save_results:
            import json
            with open(f"error_stats_{args.split}_conf{args.conf}.json", "w") as f:
                json.dump(error_stats, f, indent=4)
            print(f"💾 Статистика ошибок сохранена")
    
    # Анализ по разным порогам (опционально)
    if args.thresholds and len(args.thresholds) > 1:
        print("\n📊 Анализ по разным порогам уверенности...")
        thresholds_df = calculator.calculate_by_confidence_threshold(
            images_dir=images_dir,
            labels_dir=labels_dir,
            thresholds=args.thresholds,
            iou_threshold=args.iou
        )
        
        # Сохраняем результаты по порогам
        if args.save_results:
            thresholds_df.to_csv(f"detection_rate_by_threshold_{args.split}.csv", index=False)
            print(f"💾 Результаты по порогам сохранены")


if __name__ == "__main__":
    main()