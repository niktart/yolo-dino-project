import os
import shutil
import glob
import yaml
from typing import List, Dict, Tuple


class RandomDatasetAdder:
    def __init__(self, target_dataset_path: str, random_dataset_path: str):
        """
        Класс для добавления random датасета к основному датасету
        
        Args:
            target_dataset_path: путь к основному датасету
            random_dataset_path: путь к random датасету
        """
        self.target_dataset_path = target_dataset_path
        self.random_dataset_path = random_dataset_path
        
        # Маппинг сплитов
        self.split_mapping = {
            'train': ('train', 'train'),
            'val': ('valid', 'val'),
            'test': ('test', 'test')
        }
        
    def renumber_classes_in_random_dataset(self, start_id: int, end_id: int) -> Dict[int, int]:
        """
        Перенумеровывает классы в random датасете в указанный диапазон
        
        Args:
            start_id: начальный ID нового диапазона
            end_id: конечный ID нового диапазона
            
        Returns:
            словарь соответствия старых ID новым
        """
        print(f"🔄 Перенумерация классов random датасета в диапазон {start_id}-{end_id}...")
        
        # Находим все уникальные class_id в random датасете
        unique_ids = set()
        for split_name, (random_split, _) in self.split_mapping.items():
            labels_dir = os.path.join(self.random_dataset_path, random_split, "labels")
            if not os.path.exists(labels_dir):
                continue
                
            for filename in os.listdir(labels_dir):
                if not filename.endswith(".txt"):
                    continue
                filepath = os.path.join(labels_dir, filename)
                with open(filepath, "r") as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            unique_ids.add(class_id)

        unique_ids = sorted(list(unique_ids))
        num_classes = len(unique_ids)

        # Проверка: помещаются ли все в новый диапазон
        if num_classes > (end_id - start_id + 1):
            raise ValueError(f"Слишком много классов ({num_classes}) для диапазона {start_id}-{end_id}")

        # Создаём словарь перевода: старый ID -> новый ID
        id_map = {old_id: start_id + i for i, old_id in enumerate(unique_ids)}
        print(f"Старые ID -> новые ID:", id_map)

        # Перезаписываем файлы с новыми ID
        for split_name, (random_split, _) in self.split_mapping.items():
            labels_dir = os.path.join(self.random_dataset_path, random_split, "labels")
            if not os.path.exists(labels_dir):
                continue
                
            self._renumber_files_in_directory(labels_dir, id_map)

        print(f"✅ Перенумерация завершена!")
        return id_map
    
    def _renumber_files_in_directory(self, labels_dir: str, id_map: Dict[int, int]):
        """Перенумеровывает файлы в конкретной директории"""
        for filename in os.listdir(labels_dir):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(labels_dir, filename)
            new_lines = []
            
            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    old_class_id = int(parts[0])
                    new_class_id = id_map[old_class_id]
                    new_line = " ".join([str(new_class_id)] + parts[1:])
                    new_lines.append(new_line)
                    
            with open(filepath, "w") as f:
                f.write("\n".join(new_lines))
    
    def copy_random_to_target(self):
        """Копирует файлы из random датасета в целевой датасет"""
        print("📋 Копирование файлов из random датасета в целевой...")
        
        for split_name, (random_split, target_split) in self.split_mapping.items():
            # Пути для изображений
            random_img_dir = os.path.join(self.random_dataset_path, random_split, "images")
            target_img_dir = os.path.join(self.target_dataset_path, target_split, "images")
            
            # Пути для меток
            random_lbl_dir = os.path.join(self.random_dataset_path, random_split, "labels")
            target_lbl_dir = os.path.join(self.target_dataset_path, target_split, "labels")
            
            # Создаем целевые директории, если их нет
            os.makedirs(target_img_dir, exist_ok=True)
            os.makedirs(target_lbl_dir, exist_ok=True)
            
            # Копируем изображения
            if os.path.exists(random_img_dir):
                for img_path in glob.glob(os.path.join(random_img_dir, "*")):
                    shutil.copy(img_path, target_img_dir)
                print(f"  ✅ Скопировано изображений в {target_split}: {len(os.listdir(random_img_dir))}")
            
            # Копируем метки
            if os.path.exists(random_lbl_dir):
                for lbl_path in glob.glob(os.path.join(random_lbl_dir, "*")):
                    shutil.copy(lbl_path, target_lbl_dir)
                print(f"  ✅ Скопировано меток в {target_split}: {len(os.listdir(random_lbl_dir))}")
    
    def update_data_yaml(self, random_data_yaml_path: str):
        """
        Обновляет data.yaml целевого датасета, добавляя классы из random датасета
        
        Args:
            random_data_yaml_path: путь к data.yaml random датасета
        """
        print("📝 Обновление data.yaml...")
        
        target_yaml_path = os.path.join(self.target_dataset_path, "data.yaml")
        
        # Загружаем существующие data.yaml
        with open(target_yaml_path, "r") as f:
            target_data = yaml.safe_load(f)
        
        with open(random_data_yaml_path, "r") as f:
            random_data = yaml.safe_load(f)
        
        # Получаем списки классов
        target_names = target_data.get("names", [])
        random_names = random_data.get("names", [])
        
        # Если target_names это словарь (индекс: имя), конвертируем в список
        if isinstance(target_names, dict):
            target_names = [target_names[i] for i in sorted(target_names.keys())]
        
        # Объединяем классы
        combined_names = target_names + random_names
        
        # Подготавливаем содержимое для data.yaml
        yaml_content = {
            "path": self.target_dataset_path,
            "train": os.path.join(self.target_dataset_path, "train", "images"),
            "val": os.path.join(self.target_dataset_path, "val", "images"),
            "test": os.path.join(self.target_dataset_path, "test", "images"),
            "names": combined_names
        }
        
        # Сохраняем обновленный data.yaml
        with open(target_yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        print(f"✅ data.yaml обновлен! Теперь классов: {len(combined_names)}")
        print(f"   Было: {len(target_names)}, Добавлено: {len(random_names)}")
        
        return combined_names
    
    def run_pipeline(self, start_id: int, end_id: int):
        """
        Запускает полный пайплайн добавления random датасета
        
        Args:
            start_id: начальный ID для классов random датасета
            end_id: конечный ID для классов random датасета
        """
        print("=" * 50)
        print("➕ ДОБАВЛЕНИЕ RANDOM ДАТАСЕТА")
        print("=" * 50)
        
        # 1. Перенумеровываем классы в random датасете
        id_map = self.renumber_classes_in_random_dataset(start_id, end_id)
        
        # 2. Копируем файлы
        self.copy_random_to_target()
        
        # 3. Обновляем data.yaml
        random_yaml = os.path.join(self.random_dataset_path, "data.yaml")
        if os.path.exists(random_yaml):
            combined_names = self.update_data_yaml(random_yaml)
        else:
            print("⚠️ data.yaml random датасета не найден!")
        
        print("\n✅ Random датасет успешно добавлен!")
        print(f"📁 Итоговый датасет: {self.target_dataset_path}")
        
        return id_map