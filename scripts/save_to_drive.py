#!/usr/bin/env python
"""
Скрипт для сохранения результатов в Google Drive
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.drive_utils import DriveSaver


def parse_args():
    parser = argparse.ArgumentParser(description='Сохранение в Google Drive')
    parser.add_argument('--weights', type=str, required=True,
                        help='Путь к файлу с весами')
    parser.add_argument('--metrics', type=str, required=True,
                        help='Путь к файлу с метриками')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Название эксперимента')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("💾 СОХРАНЕНИЕ В GOOGLE DRIVE")
    print("=" * 50)
    
    # Проверяем существование файлов
    if not os.path.exists(args.weights):
        print(f"❌ Файл с весами не найден: {args.weights}")
        return
    
    if not os.path.exists(args.metrics):
        print(f"❌ Файл с метриками не найден: {args.metrics}")
        return
    
    # Сохраняем в Drive
    drive_saver = DriveSaver()
    drive_saver.mount_drive()
    drive_saver.save_experiment_results(
        weights_path=args.weights,
        metrics_path=args.metrics,
        experiment_name=args.exp_name
    )
    
    print("\n✅ Все файлы успешно сохранены в Google Drive!")


if __name__ == "__main__":
    main()