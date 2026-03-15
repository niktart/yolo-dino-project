import os
import json
from PIL import Image
from tqdm import tqdm
import yaml
from typing import Dict, List, Any


class YOLOToCOCOConverter:
    """Конвертер датасета из формата YOLO в COCO"""
    
    def __init__(self, dataset_root: str, splits: List[str] = None):
        """
        Args:
            dataset_root: корневая папка датасета
            splits: список сплитов для конвертации
        """
        self.dataset_root = dataset_root
        self.splits = splits or ["train", "val", "test"]
        self.class_names = []
        
    def load_class_names(self) -> List[str]:
        """Загружает имена классов из data.yaml"""
        yaml_path = os.path.join(self.dataset_root, "data.yaml")
        with open(yaml_path) as f:
            data_yaml = yaml.safe_load(f)
        
        names = data_yaml["names"]
        # Если names это словарь, конвертируем в список
        if isinstance(names, dict):
            self.class_names = [names[i] for i in sorted(names.keys())]
        else:
            self.class_names = names
        
        return self.class_names
    
    def yolo_to_coco_bbox(self, xc: float, yc: float, w: float, h: float, 
                          img_w: int, img_h: int) -> List[float]:
        """
        Конвертирует bbox из формата YOLO (нормализованный) в COCO (пиксели)
        
        Args:
            xc, yc: центр объекта (нормализовано)
            w, h: ширина и высота (нормализовано)
            img_w, img_h: размеры изображения в пикселях
            
        Returns:
            bbox в формате COCO: [x_min, y_min, width, height]
        """
        x_min = (xc - w / 2) * img_w
        y_min = (yc - h / 2) * img_h
        width = w * img_w
        height = h * img_h
        return [x_min, y_min, width, height]
    
    def convert_split(self, split_name: str) -> Dict[str, Any]:
        """
        Конвертирует один сплит в COCO формат
        
        Args:
            split_name: название сплита ('train', 'val', 'test')
            
        Returns:
            словарь в формате COCO
        """
        images = []
        annotations = []

        image_dir = os.path.join(self.dataset_root, split_name, "images")
        label_dir = os.path.join(self.dataset_root, split_name, "labels")

        if not os.path.exists(image_dir):
            print(f"⚠️ Директория не найдена: {image_dir}")
            return None

        ann_id = 1
        img_id = 1

        # Проходим по всем изображениям
        for img_file in tqdm(sorted(os.listdir(image_dir)), 
                            desc=f"Processing {split_name}"):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

            try:
                width, height = Image.open(img_path).size
            except Exception as e:
                print(f"Ошибка при открытии {img_path}: {e}")
                continue

            # Информация об изображении
            images.append({
                "id": img_id,
                "file_name": img_file,
                "width": width,
                "height": height
            })

            # Читаем аннотации
            if os.path.exists(label_path):
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        cls = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])

                        bbox = self.yolo_to_coco_bbox(xc, yc, w, h, width, height)

                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls + 1,  # COCO id начинается с 1
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        })
                        ann_id += 1

            img_id += 1

        # Категории
        categories = [
            {
                "id": i + 1,
                "name": str(name)  # Обязательно строка для Detectron2
            }
            for i, name in enumerate(self.class_names)
        ]

        return {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
    
    def convert_all(self, output_dir: str = None) -> Dict[str, str]:
        """
        Конвертирует все сплиты и сохраняет JSON файлы
        
        Args:
            output_dir: директория для сохранения (если None, используется dataset_root)
            
        Returns:
            словарь: split_name -> path_to_json
        """
        if output_dir is None:
            output_dir = self.dataset_root
        
        # Загружаем имена классов
        self.load_class_names()
        print(f"📋 Загружено классов: {len(self.class_names)}")
        
        results = {}
        
        for split in self.splits:
            split_path = os.path.join(self.dataset_root, split)
            if not os.path.isdir(split_path):
                print(f"⚠️ Сплит {split} не найден, пропускаем")
                continue

            coco_dict = self.convert_split(split)
            
            if coco_dict is None:
                continue
                
            output_path = os.path.join(output_dir, f"{split}_coco.json")
            with open(output_path, "w") as f:
                json.dump(coco_dict, f, indent=2)

            print(f"✅ Сохранен {output_path}")
            results[split] = output_path
        
        return results