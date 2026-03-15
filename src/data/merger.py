import os
import shutil
import yaml
import pandas as pd
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


class DatasetMerger:
    def __init__(self, source_dir, target_dir, rename_map, split_ratios, random_state):
        """
        Args:
            source_dir: путь к папке с исходными датасетами (для чтения)
            target_dir: путь для сохранения объединенного датасета (будет создан)
            rename_map: маппинг переименования классов
            split_ratios: пропорции сплитов
            random_state: random state
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.rename_map = rename_map
        self.split_ratios = split_ratios
        self.random_state = random_state
        
        self.dataset_class_maps = {}
        self.final_names = []
        self.final_map = {}
        self.stats = {}
        
    def collect_classes(self):
        """Собирает классы из всех исходных датасетов"""
        print(f"📥 Сбор классов из {self.source_dir}")
        
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"Папка не найдена: {self.source_dir}")
        
        found_datasets = 0
        for ds in os.listdir(self.source_dir):
            ds_path = os.path.join(self.source_dir, ds)
            if not os.path.isdir(ds_path):
                continue
                
            yaml_path = os.path.join(ds_path, "data.yaml")
            if not os.path.isfile(yaml_path):
                continue

            try:
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)

                # Получаем имена классов
                class_names = data.get("names", [])
                if isinstance(class_names, dict):
                    class_names = [class_names[i] for i in sorted(class_names.keys())]
                
                # Применяем rename_map
                renamed_classes = [
                    self.rename_map.get(name, name).lower()
                    for name in class_names
                ]
                
                self.dataset_class_maps[ds] = renamed_classes
                found_datasets += 1
                print(f"  ✅ {ds}: {len(renamed_classes)} классов")
                
            except Exception as e:
                print(f"  ⚠️ Ошибка при чтении {ds}: {e}")
                continue
        
        print(f"📊 Найдено датасетов: {found_datasets}")
        return self.dataset_class_maps
    
    def collect_samples(self):
        """Собирает все изображения и аннотации из train сплитов"""
        samples = []  # (img_path, objects)
        all_classes = set()

        for ds, class_list in self.dataset_class_maps.items():
            ds_path = os.path.join(self.source_dir, ds)
            
            # Ищем train папку
            train_dir = None
            for possible_train in ["train", "Train", "training", "TRAIN"]:
                if os.path.exists(os.path.join(ds_path, possible_train)):
                    train_dir = possible_train
                    break
            
            if train_dir is None:
                print(f"⚠️ В датасете {ds} нет train папки, пропускаем")
                continue
            
            img_dir = os.path.join(ds_path, train_dir, "images")
            lbl_dir = os.path.join(ds_path, train_dir, "labels")
            
            if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
                print(f"⚠️ В датасете {ds} нет images или labels, пропускаем")
                continue

            # Собираем все аннотации
            for lbl_file in os.listdir(lbl_dir):
                if not lbl_file.endswith(".txt"):
                    continue

                lbl_path = os.path.join(lbl_dir, lbl_file)
                
                # Ищем изображение
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_img = os.path.join(img_dir, lbl_file.replace(".txt", ext))
                    if os.path.exists(potential_img):
                        img_path = potential_img
                        break
                
                if img_path is None:
                    continue

                # Читаем аннотации
                try:
                    with open(lbl_path) as f:
                        lines = f.readlines()
                except:
                    continue

                objects = []
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(float(parts[0]))
                        cls_name = class_list[cls_id]
                        all_classes.add(cls_name)
                        objects.append((cls_name, parts[1:]))
                    except:
                        continue

                if objects:
                    samples.append((img_path, objects))

        self.final_names = sorted(all_classes)
        self.final_map = {cls: i for i, cls in enumerate(self.final_names)}
        
        print(f"📸 Всего изображений: {len(samples)}")
        print(f"📊 Всего классов: {len(self.final_names)}")
        return samples
    
    def create_multilabel_matrix(self, samples):
        """Создает мультиклассовую матрицу для стратификации"""
        X = []
        y = []
        
        for img_path, objects in samples:
            X.append((img_path, objects))
            label_vector = [0] * len(self.final_names)
            for cls_name, _ in objects:
                label_vector[self.final_map[cls_name]] = 1
            y.append(label_vector)
        
        return X, y
    
    def split_data(self, X, y):
        """Стратифицированное разбиение данных"""
        if len(X) == 0:
            raise ValueError("Нет данных для разбиения!")
        
        # Первый сплит: train / temp
        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, 
            test_size=1 - self.split_ratios["train"], 
            random_state=self.random_state
        )
        train_idx, temp_idx = next(msss1.split(X, y))
        
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_temp = [X[i] for i in temp_idx]
        y_temp = [y[i] for i in temp_idx]

        # Второй сплит: val / test
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=self.split_ratios["test"] / (self.split_ratios["val"] + self.split_ratios["test"]),
            random_state=self.random_state
        )
        val_idx, test_idx = next(msss2.split(X_temp, y_temp))
        
        X_val = [X_temp[i] for i in val_idx]
        X_test = [X_temp[i] for i in test_idx]
        
        splits = {"train": X_train, "val": X_val, "test": X_test}
        
        print(f"✂️ Разбиение:")
        print(f"  train: {len(X_train)} изображений")
        print(f"  val: {len(X_val)} изображений")
        print(f"  test: {len(X_test)} изображений")
        
        return splits
    
    def copy_files_with_remap(self, splits):
        """Копирует файлы с перемаппингом классов"""
        # Создаем целевую папку
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Инициализация статистики
        self.stats = {
            split: {cls: {"images": set(), "bbox": 0} for cls in self.final_names} 
            for split in splits
        }

        for split, split_samples in splits.items():
            img_out = os.path.join(self.target_dir, split, "images")
            lbl_out = os.path.join(self.target_dir, split, "labels")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)

            for img_path, objects in tqdm(split_samples, desc=f"Copy {split}"):
                # Генерируем уникальное имя
                base_name = os.path.basename(img_path)
                name = f"{split}_{len(os.listdir(img_out))}_{base_name}".replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
                
                new_lines = []
                image_classes = set()

                for cls_name, bbox in objects:
                    cls_id = self.final_map[cls_name]
                    new_lines.append(" ".join([str(cls_id)] + bbox))
                    self.stats[split][cls_name]["bbox"] += 1
                    image_classes.add(cls_name)

                for cls_name in image_classes:
                    self.stats[split][cls_name]["images"].add(name)

                # Копируем изображение
                dst_img_path = os.path.join(img_out, f"{name}.jpg")
                shutil.copy(img_path, dst_img_path)
                
                # Сохраняем аннотации
                with open(os.path.join(lbl_out, f"{name}.txt"), "w") as f:
                    f.write("\n".join(new_lines) + "\n")
    
    def save_data_yaml(self):
        """Сохраняет data.yaml"""
        data_yaml = {
            "path": self.target_dir,
            "train": os.path.join("train", "images"),
            "val": os.path.join("val", "images"),
            "test": os.path.join("test", "images"),
            "names": self.final_names,
        }

        with open(os.path.join(self.target_dir, "data.yaml"), "w") as f:
            yaml.dump(data_yaml, f, sort_keys=False)

        print(f"✅ data.yaml создан в {self.target_dir}")
    
    def print_statistics(self):
        """Выводит статистику по датасету"""
        rows = []
        for cls in self.final_names:
            row = {"class_name": cls}
            total_bbox = 0
            total_images = 0
            for split in ["train", "val", "test"]:
                bbox = self.stats[split][cls]["bbox"]
                images = len(self.stats[split][cls]["images"])
                row[f"{split}_bbox"] = bbox
                row[f"{split}_images"] = images
                total_bbox += bbox
                total_images += images
            row["total_bbox"] = total_bbox
            row["total_images"] = total_images
            rows.append(row)

        df_stats = pd.DataFrame(rows).sort_values("total_bbox", ascending=False).reset_index(drop=True)
        print("📊 Финальная статистика:")
        print(df_stats)
        return df_stats
    
    def run_pipeline(self):
        """Запускает полный пайплайн объединения"""
        print("📥 Сбор классов из исходных датасетов...")
        self.collect_classes()
        
        print("\n📸 Сбор образцов из train сплитов...")
        samples = self.collect_samples()
        
        if len(samples) == 0:
            raise ValueError("Не найдено ни одного образца для объединения!")
        
        print("\n🔢 Создание мультиклассовой матрицы...")
        X, y = self.create_multilabel_matrix(samples)
        
        print("\n✂️ Стратифицированное разбиение...")
        splits = self.split_data(X, y)
        
        print("\n📋 Копирование файлов в целевой датасет...")
        self.copy_files_with_remap(splits)
        
        print("\n📝 Сохранение data.yaml...")
        self.save_data_yaml()
        
        print("\n📊 Вывод статистики...")
        df = self.print_statistics()
        
        return df