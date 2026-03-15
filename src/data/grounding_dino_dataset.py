import os
import csv
import random
from collections import defaultdict

class GroundingDINODataset:
    def __init__(self, ann_file, images_dir, split='train'):
        self.ann_file = ann_file
        self.images_dir = images_dir
        self.split = split
        self.data = self._load()
        
    def _load(self):
        """Загружает датасет из CSV"""
        ann_dict = defaultdict(lambda: defaultdict(list))
        
        print(f"📥 Загрузка {self.split} датасета из {self.ann_file}")
        
        with open(self.ann_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(self.images_dir, row['image_name'])
                
                if not os.path.exists(img_path):
                    continue
                
                # Конвертируем bbox
                x1 = int(row['bbox_x'])
                y1 = int(row['bbox_y'])
                w = int(row['bbox_width'])
                h = int(row['bbox_height'])
                x2 = x1 + w
                y2 = y1 + h
                
                ann_dict[img_path]['boxes'].append([x1, y1, x2, y2])
                ann_dict[img_path]['captions'].append(row['label_name'])
        
        print(f"✅ Загружено {len(ann_dict)} изображений")
        return dict(ann_dict)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        items = list(self.data.items())
        return items[idx]
    
    def get_all_items(self):
        return list(self.data.items())