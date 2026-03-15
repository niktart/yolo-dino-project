import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
from collections import defaultdict
import yaml
from typing import Dict, Set, List, Tuple


class DatasetAugmenter:
    def __init__(self, dataset_path: str, min_images_per_class: int = 150):
        """
        Класс для аугментации датасета
        
        Args:
            dataset_path: путь к датасету
            min_images_per_class: минимальное количество изображений на класс
        """
        self.dataset_path = dataset_path
        self.min_images_per_class = min_images_per_class
        
        self.train_img_dir = os.path.join(dataset_path, "train", "images")
        self.train_lbl_dir = os.path.join(dataset_path, "train", "labels")
        
        # Определяем аугментации
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        self.class_names = []
        self.class_images = defaultdict(set)
        
    def load_class_names(self) -> List[str]:
        """Загружает названия классов из data.yaml"""
        yaml_path = os.path.join(self.dataset_path, "data.yaml")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            names = data.get('names', [])
            
            # Если names это словарь, конвертируем в список
            if isinstance(names, dict):
                self.class_names = [names[i] for i in sorted(names.keys())]
            else:
                self.class_names = names
                
        return self.class_names
    
    def count_images_per_class(self) -> Dict[int, Set[str]]:
        """
        Подсчитывает количество изображений для каждого класса
        Returns:
            словарь: class_id -> множество путей к изображениям
        """
        print("📊 Подсчет изображений по классам...")
        
        if not os.path.exists(self.train_lbl_dir):
            raise FileNotFoundError(f"Директория не найдена: {self.train_lbl_dir}")
        
        for lbl_file in os.listdir(self.train_lbl_dir):
            if not lbl_file.endswith('.txt'):
                continue

            img_file = lbl_file.replace('.txt', '.jpg')
            img_path = os.path.join(self.train_img_dir, img_file)

            if not os.path.exists(img_path):
                continue
                
            with open(os.path.join(self.train_lbl_dir, lbl_file)) as f:
                for line in f:
                    if line.strip():
                        cls_id = int(float(line.split()[0]))  # безопасно для "3.0"
                        self.class_images[cls_id].add(img_path)
        
        return self.class_images
    
    def collect_samples_for_class(self, cls_id: int) -> List[Tuple[str, List[List[float]], List[int]]]:
        """
        Собирает все изображения с заданным классом и соответствующие bbox
        
        Args:
            cls_id: ID класса
            
        Returns:
            список кортежей (путь_к_изображению, список_bbox, список_классов)
        """
        samples = []
        
        for img_path in self.class_images[cls_id]:
            lbl_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            lbl_path = os.path.join(self.train_lbl_dir, lbl_name)

            if not os.path.exists(lbl_path):
                continue

            bboxes = []
            classes = []
            
            with open(lbl_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    cls_in_file = int(float(parts[0]))
                    if cls_in_file == cls_id:
                        bboxes.append([float(x) for x in parts[1:5]])
                        classes.append(cls_id)

            if bboxes:
                samples.append((img_path, bboxes, classes))
                
        return samples
    
    def _clip_bbox(self, bbox: List[float]) -> List[float]:
        """Ограничивает bbox значениями [0, 1]"""
        return np.clip(bbox, 0, 1).tolist()
    
    def _is_valid_bbox(self, bbox: List[float]) -> bool:
        """Проверяет, что bbox имеет положительную площадь"""
        x, y, w, h = bbox
        return w > 0 and h > 0
    
    def augment_class(self, cls_id: int, needed_count: int) -> int:
        """
        Аугментирует изображения для конкретного класса
        
        Args:
            cls_id: ID класса
            needed_count: сколько аугментаций нужно создать
            
        Returns:
            количество созданных аугментаций
        """
        cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
        print(f"  ⚠️ {cls_name}: нужно {needed_count} аугментаций")
        
        # Собираем все изображения с этим классом
        class_samples = self.collect_samples_for_class(cls_id)
        
        if not class_samples:
            print(f"  ⚠️ Нет изображений для класса {cls_name}!")
            return 0
        
        aug_count = 0
        attempts = 0
        max_attempts = needed_count * 10  # Защита от бесконечного цикла
        
        while aug_count < needed_count and attempts < max_attempts:
            attempts += 1
            random.shuffle(class_samples)
            
            for img_path, bboxes, classes in class_samples:
                if aug_count >= needed_count:
                    break

                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                try:
                    augmented = self.transform(
                        image=img, 
                        bboxes=bboxes, 
                        class_labels=classes
                    )

                    # Обрабатываем bbox
                    safe_bboxes = []
                    safe_classes = []
                    
                    for bbox, cls in zip(augmented['bboxes'], augmented['class_labels']):
                        clipped_bbox = self._clip_bbox(bbox)
                        if self._is_valid_bbox(clipped_bbox):
                            safe_bboxes.append(clipped_bbox)
                            safe_classes.append(cls)

                    if not safe_bboxes:
                        continue

                    # Сохраняем изображение
                    aug_name = f"aug_{cls_name}_{aug_count}_{os.path.basename(img_path)}"
                    aug_img_path = os.path.join(self.train_img_dir, aug_name)
                    cv2.imwrite(
                        aug_img_path, 
                        cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                    )

                    # Сохраняем метки
                    lbl_name = os.path.splitext(aug_name)[0] + ".txt"
                    lbl_path = os.path.join(self.train_lbl_dir, lbl_name)
                    
                    with open(lbl_path, 'w') as f:
                        for bbox, cls in zip(safe_bboxes, safe_classes):
                            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

                    aug_count += 1

                except Exception as e:
                    print(f"    Ошибка аугментации: {e}")
                    continue
        
        return aug_count
    
    def run_pipeline(self) -> Dict[int, int]:
        """
        Запускает полный пайплайн аугментации
        Returns:
            словарь: class_id -> количество созданных аугментаций
        """
        print("=" * 50)
        print("🔄 АУГМЕНТАЦИЯ ДАТАСЕТА")
        print("=" * 50)
        
        # 1. Загружаем классы
        self.load_class_names()
        print(f"📋 Загружено классов: {len(self.class_names)}")
        
        # 2. Подсчитываем изображения
        self.count_images_per_class()
        
        # 3. Аугментируем классы с недостаточным количеством
        augmentation_stats = {}
        
        for cls_id, cls_name in enumerate(self.class_names):
            current_count = len(self.class_images[cls_id])
            needed = max(0, self.min_images_per_class - current_count)
            
            if needed <= 0:
                print(f"✅ {cls_name}: {current_count} изображений")
                augmentation_stats[cls_id] = 0
                continue
            
            created = self.augment_class(cls_id, needed)
            augmentation_stats[cls_id] = created
            
            if created >= needed:
                print(f"  ✅ Создано {created} аугментаций для {cls_name}")
            else:
                print(f"  ⚠️ Создано только {created} из {needed} для {cls_name}")
        
        # 4. Итоговая статистика
        total_created = sum(augmentation_stats.values())
        print(f"\n📊 ИТОГО: создано {total_created} аугментаций")
        
        return augmentation_stats