import os
import json
import csv
from typing import Dict, List


class COKOToCSVConverter:
    """Конвертер из COCO формата в CSV для Grounding DINO"""
    
    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: корневая папка датасета
        """
        self.dataset_root = dataset_root
        
    def convert(self, coco_json_path: str, output_csv_path: str) -> str:
        """
        Конвертирует COCO JSON в CSV
        
        Args:
            coco_json_path: путь к COCO JSON файлу
            output_csv_path: путь для сохранения CSV
            
        Returns:
            путь к сохраненному CSV
        """
        print(f"🔄 Конвертация {coco_json_path} -> {output_csv_path}")
        
        with open(coco_json_path) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        with open(output_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "label_name",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "image_name"
            ])

            for ann in coco["annotations"]:
                image_info = images[ann["image_id"]]
                category_name = categories[ann["category_id"]]

                x, y, w, h = ann["bbox"]

                writer.writerow([
                    category_name,
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                    image_info["file_name"]
                ])

        print(f"✅ CSV сохранен: {output_csv_path}")
        return output_csv_path
    
    def convert_all_splits(self, splits: List[str] = None) -> Dict[str, str]:
        """
        Конвертирует все сплиты
        
        Args:
            splits: список сплитов для конвертации
            
        Returns:
            словарь: split_name -> path_to_csv
        """
        if splits is None:
            splits = ["train", "val", "test"]
        
        results = {}
        
        for split in splits:
            json_path = os.path.join(self.dataset_root, f"{split}_coco.json")
            if not os.path.exists(json_path):
                print(f"⚠️ {json_path} не найден, пропускаем")
                continue
                
            csv_path = os.path.join(self.dataset_root, f"{split}_annotations.csv")
            self.convert(json_path, csv_path)
            results[split] = csv_path
        
        return results