import os
import yaml
from collections import defaultdict
from difflib import SequenceMatcher
import random
import matplotlib.pyplot as plt
import cv2


class ClassAnalyzer:
    def __init__(self, data_path, similarity_threshold=0.8):
        """
        Args:
            data_path: путь к папке с исходными датасетами (где лежат все data.yaml)
        """
        self.data_path = data_path
        self.similarity_threshold = similarity_threshold
        self.all_classes = set()
        self.dataset_classes = {}
        self.substring_groups = {}
        
    def collect_all_classes(self):
        """Собирает все классы из всех датасетов в data_path"""
        print(f"🔍 Поиск датасетов в {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Папка не найдена: {self.data_path}")
        
        # Собираем все папки, в которых есть data.yaml
        found_datasets = 0
        for ds in os.listdir(self.data_path):
            ds_path = os.path.join(self.data_path, ds)
            if not os.path.isdir(ds_path):
                continue
                
            yaml_path = os.path.join(ds_path, "data.yaml")
            if not os.path.isfile(yaml_path):
                continue

            try:
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)

                class_list = data.get("names", [])
                if isinstance(class_list, dict):
                    class_list = [class_list[i] for i in sorted(class_list.keys())]
                
                class_list = [cls.strip() for cls in class_list]
                self.dataset_classes[ds] = class_list
                
                for cls in class_list:
                    self.all_classes.add(cls)
                
                found_datasets += 1
                print(f"  ✅ {ds}: {len(class_list)} классов")
                
            except Exception as e:
                print(f"  ⚠️ Ошибка при чтении {ds}: {e}")
                continue

        print(f"\n📊 Найдено датасетов: {found_datasets}")
        self.all_classes = sorted(self.all_classes)
        print(f"📊 Всего уникальных классов: {len(self.all_classes)}")
        return self.all_classes
    
    def normalize(self, s):
        """Нормализует строку для сравнения"""
        return s.lower().replace("-", " ").replace("_", " ").strip()
    
    def find_conflicting_classes(self):
        """Находит похожие/конфликтующие классы"""
        normalized_classes = {cls: self.normalize(cls) for cls in self.all_classes}
        substring_groups = defaultdict(set)

        for cls_a, norm_a in normalized_classes.items():
            for cls_b, norm_b in normalized_classes.items():
                if cls_a == cls_b:
                    continue

                if len(norm_a) < 4 or len(norm_b) < 4:
                    continue

                # Проверка на подстроки
                if norm_a in norm_b or norm_b in norm_a:
                    substring_groups[norm_a].add(cls_a)
                    substring_groups[norm_a].add(cls_b)
                    continue

                # Проверка на похожесть
                similarity = SequenceMatcher(None, norm_a, norm_b).ratio()
                if similarity >= self.similarity_threshold:
                    substring_groups[norm_a].add(cls_a)
                    substring_groups[norm_a].add(cls_b)

        # Оставляем только группы с >1 класса
        self.substring_groups = {
            k: sorted(v)
            for k, v in substring_groups.items()
            if len(v) > 1
        }
        
        print(f"\n🔎 Найдено возможных опечаток / похожих названий: {len(self.substring_groups)}")
        for base, classes in self.substring_groups.items():
            print(f"  '{base}': {classes}")
        
        return self.substring_groups
    
    def collect_images_per_class(self):
        """Собирает пути к изображениям по классам (только из train сплита)"""
        images_per_class = defaultdict(list)
        
        # ОТЛАДКА: считаем статистику по каждому датасету
        total_images_found = 0
        total_annotations_found = 0

        for ds, classes in self.dataset_classes.items():
            ds_path = os.path.join(self.data_path, ds)
            print(f"\n📁 Обработка датасета: {ds}")
            
            # Ищем train папку
            train_dir = None
            for possible_train in ["train", "Train", "training", "TRAIN"]:
                train_candidate = os.path.join(ds_path, possible_train)
                if os.path.exists(train_candidate):
                    train_dir = possible_train
                    print(f"  ✅ Найдена train папка: {possible_train}")
                    break
            
            if train_dir is None:
                print(f"  ⚠️ В датасете {ds} нет train папки, пропускаем")
                continue
                
            img_dir = os.path.join(ds_path, train_dir, "images")
            lbl_dir = os.path.join(ds_path, train_dir, "labels")
            
            print(f"  📂 images: {img_dir}")
            print(f"  📂 labels: {lbl_dir}")

            if not os.path.exists(img_dir):
                print(f"  ❌ Нет папки images: {img_dir}")
                continue
                
            if not os.path.exists(lbl_dir):
                print(f"  ❌ Нет папки labels: {lbl_dir}")
                continue

            # Считаем количество файлов
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            lbl_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
            print(f"  📊 Найдено изображений: {len(img_files)}")
            print(f"  📊 Найдено аннотаций: {len(lbl_files)}")

            # Собираем все аннотации
            ds_images_found = 0
            ds_annotations_found = 0
            
            for lbl_file in lbl_files:
                lbl_path = os.path.join(lbl_dir, lbl_file)
                
                # Пробуем найти изображение с разными расширениями
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_img = os.path.join(img_dir, lbl_file.replace(".txt", ext))
                    if os.path.exists(potential_img):
                        img_path = potential_img
                        break
                
                if img_path is None:
                    continue
                
                ds_images_found += 1

                # Читаем аннотации
                try:
                    with open(lbl_path) as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"    ⚠️ Ошибка чтения {lbl_file}: {e}")
                    continue

                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        parts = line.strip().split()
                        cls_id = int(float(parts[0]))
                        if cls_id >= len(classes):
                            print(f"    ⚠️ class_id {cls_id} больше чем классов {len(classes)} в {lbl_file}")
                            continue
                        cls_name = classes[cls_id]
                        images_per_class[cls_name].append((img_path, ds))
                        ds_annotations_found += 1
                    except Exception as e:
                        print(f"    ⚠️ Ошибка парсинга строки '{line.strip()}': {e}")
                        continue
            
            print(f"  📊 В датасете {ds} обработано:")
            print(f"    - изображений с аннотациями: {ds_images_found}")
            print(f"    - всего аннотаций: {ds_annotations_found}")
            
            total_images_found += ds_images_found
            total_annotations_found += ds_annotations_found

        print(f"\n📊 ВСЕГО ПО ВСЕМ ДАТАСЕТАМ:")
        print(f"  - изображений с аннотациями: {total_images_found}")
        print(f"  - всего аннотаций: {total_annotations_found}")
        
        # Статистика по классам
        print(f"\n📊 ТОП-10 КЛАССОВ ПО КОЛИЧЕСТВУ ИЗОБРАЖЕНИЙ:")
        class_counts = {cls: len(imgs) for cls, imgs in images_per_class.items()}
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_classes[:10]:
            print(f"  {cls}: {count} изображений")
        
        return images_per_class
    
    def visualize_conflicts(self, images_per_class, images_per_class_count=6):
        """Визуализирует конфликтующие классы"""
        if not self.substring_groups:
            print("📊 Нет конфликтующих классов для визуализации")
            return
        
        # ОТЛАДКА: проверяем наличие изображений для конфликтующих классов
        print("\n🔍 ПРОВЕРКА НАЛИЧИЯ ИЗОБРАЖЕНИЙ ДЛЯ КОНФЛИКТУЮЩИХ КЛАССОВ:")
        for base, classes in self.substring_groups.items():
            print(f"\n🧩 Группа '{base}':")
            for cls_name in classes:
                count = len(images_per_class.get(cls_name, []))
                print(f"  {cls_name}: {count} изображений")
            
        for base, classes in self.substring_groups.items():
            print(f"\n🧩 СПОРНАЯ ГРУППА: '{base}'")

            total_classes = len(classes)
            if total_classes == 0:
                continue
            
            # Проверяем, есть ли хоть одно изображение для любого класса в группе
            has_any_images = any(len(images_per_class.get(cls_name, [])) > 0 for cls_name in classes)
            if not has_any_images:
                print(f"  ⚠️ Нет изображений ни для одного класса в группе '{base}', пропускаем визуализацию")
                continue
                
            plt.figure(figsize=(16, 4 * total_classes))

            for row_idx, cls_name in enumerate(classes):
                imgs = images_per_class.get(cls_name, [])
                if not imgs:
                    print(f"  ⚠️ Нет изображений для класса: {cls_name}")
                    # Добавляем пустой subplot
                    for col_idx in range(images_per_class_count):
                        plt.subplot(total_classes, images_per_class_count, 
                                   row_idx * images_per_class_count + col_idx + 1)
                        plt.text(0.5, 0.5, f"Нет изображений\nдля {cls_name}", 
                                ha='center', va='center', fontsize=8)
                        plt.axis("off")
                    continue

                print(f"  ✅ Для класса {cls_name} найдено {len(imgs)} изображений")
                
                # Берем случайные изображения
                try:
                    sample_imgs = random.sample(imgs, min(images_per_class_count, len(imgs)))
                except ValueError as e:
                    print(f"  ⚠️ Ошибка при выборке изображений для {cls_name}: {e}")
                    continue

                for col_idx, (img_path, ds_name) in enumerate(sample_imgs):
                    plt.subplot(total_classes, images_per_class_count, 
                               row_idx * images_per_class_count + col_idx + 1)
                    try:
                        # Проверяем существование файла
                        if not os.path.exists(img_path):
                            print(f"    ❌ Файл не существует: {img_path}")
                            plt.text(0.5, 0.5, f"Файл\nне найден", ha='center', va='center')
                        else:
                            img = cv2.imread(img_path)
                            if img is None:
                                print(f"    ❌ Не удалось прочитать: {img_path}")
                                plt.text(0.5, 0.5, f"Ошибка\nчтения", ha='center', va='center')
                            else:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                plt.imshow(img)
                                plt.title(f"{cls_name}\n{ds_name}", fontsize=8)
                    except Exception as e:
                        print(f"    ❌ Ошибка при отображении {img_path}: {e}")
                        plt.text(0.5, 0.5, f"Ошибка\n{e}", ha='center', va='center')
                    plt.axis("off")

            plt.tight_layout()
            plt.show()
            print(f"  ✅ Визуализация для группы '{base}' завершена")