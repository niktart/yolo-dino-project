import os
import shutil
from pathlib import Path


class DriveSaver:
    """Класс для сохранения результатов в Google Drive"""
    
    def __init__(self, base_drive_path="/content/drive/MyDrive/term_work"):
        self.base_drive_path = base_drive_path
        self.weights_path = os.path.join(base_drive_path, "compare", "weights")
        self.metrics_path = os.path.join(base_drive_path, "compare", "metrics")
        
    def mount_drive(self):
        """Монтирует Google Drive"""
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
        
    def ensure_dirs(self):
        """Создает необходимые директории"""
        os.makedirs(self.weights_path, exist_ok=True)
        os.makedirs(self.metrics_path, exist_ok=True)
        print(f"📁 Директории созданы:\n  {self.weights_path}\n  {self.metrics_path}")
        
    def save_model_weights(self, source_path, experiment_name, suffix=""):
        """
        Сохраняет веса модели в Google Drive
        
        Args:
            source_path: путь к файлу с весами
            experiment_name: название эксперимента
            suffix: дополнительный суффикс для файла
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Файл не найден: {source_path}")
        
        # Формируем имя файла
        if suffix:
            filename = f"{experiment_name}_{suffix}.pt"
        else:
            filename = f"{experiment_name}.pt"
            
        dest_path = os.path.join(self.weights_path, filename)
        
        # Копируем файл
        shutil.copy(source_path, dest_path)
        print(f"✅ Веса модели сохранены: {dest_path}")
        
        return dest_path
    
    def save_metrics(self, source_path, experiment_name, suffix=""):
        """
        Сохраняет метрики в Google Drive
        
        Args:
            source_path: путь к файлу с метриками
            experiment_name: название эксперимента
            suffix: дополнительный суффикс для файла
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Файл не найден: {source_path}")
        
        # Формируем имя файла
        if suffix:
            filename = f"{experiment_name}_{suffix}.json"
        else:
            filename = f"{experiment_name}.json"
            
        dest_path = os.path.join(self.metrics_path, filename)
        
        # Копируем файл
        shutil.copy(source_path, dest_path)
        print(f"✅ Метрики сохранены: {dest_path}")
        
        return dest_path
    
    def save_experiment_results(self, weights_path, metrics_path, experiment_name):
        """
        Сохраняет все результаты эксперимента
        
        Args:
            weights_path: путь к файлу с весами
            metrics_path: путь к файлу с метриками
            experiment_name: название эксперимента
        """
        self.ensure_dirs()
        
        weights_dest = self.save_model_weights(weights_path, experiment_name)
        metrics_dest = self.save_metrics(metrics_path, experiment_name)
        
        print(f"\n✅ Эксперимент '{experiment_name}' полностью сохранен в Drive")
        
        return weights_dest, metrics_dest