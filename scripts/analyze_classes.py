#!/usr/bin/env python
"""
Скрипт для анализа классов в исходных датасетах
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.data.class_analyzer import ClassAnalyzer


def main():
    print("=" * 50)
    print("🔍 АНАЛИЗ КЛАССОВ")
    print("=" * 50)
    
    # Проверяем наличие папки с исходными датасетами
    if not os.path.exists(settings.SOURCE_DATASETS_PATH):
        print(f"❌ Папка с датасетами не найдена: {settings.SOURCE_DATASETS_PATH}")
        print("Сначала запустите: python scripts/download_datasets.py")
        return
    
    # Создаем анализатор для исходных датасетов
    analyzer = ClassAnalyzer(settings.SOURCE_DATASETS_PATH)
    
    # Собираем все классы
    print("\n📊 Сбор всех классов из исходных датасетов...")
    all_classes = analyzer.collect_all_classes()
    
    # Ищем конфликтующие классы
    print("\n🔎 Поиск конфликтующих классов...")
    conflicts = analyzer.find_conflicting_classes()
    
    # Если есть конфликты, собираем изображения и визуализируем
    if conflicts:
        print("\n🖼️ Сбор изображений для визуализации...")
        images_per_class = analyzer.collect_images_per_class()
        
        print("\n📈 Визуализация конфликтов...")
        analyzer.visualize_conflicts(images_per_class, images_per_class_count=6)
    else:
        print("\n✅ Конфликтов не найдено! Можно переходить к объединению.")
    
    print("\n✅ Анализ завершен!")


if __name__ == "__main__":
    main()