import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

from ultralytics import YOLOWorld

from src.utils.timer import Timer
from src.evaluation.metrics import MetricsCalculator


class YOLOWorldModel:
    """Класс для работы с моделью YOLO-World"""
    
    def __init__(self, model_path: str = "yolov8s-worldv2.pt"):
        """
        Инициализация модели YOLO-World
        
        Args:
            model_path: путь к весам модели или название модели
        """
        self.model_path = model_path
        self.model = None
        self.metrics_calculator = MetricsCalculator()
        
    def load_model(self):
        """Загружает модель"""
        print(f"📥 Загрузка модели {self.model_path}...")
        self.model = YOLOWorld(self.model_path)
        print("✅ Модель загружена")
        return self.model
    
    def train(self, 
              data_yaml: str,
              epochs: int = 3,
              imgsz: int = 640,
              batch: int = 16,
              lr0: float = 0.001,
              lrf: float = 0.01,
              cos_lr: bool = True,
              freeze: int = -5,
              mixup: float = 0.1,
              workers: int = 0,
              cache: bool = False,
              **kwargs) -> Dict[str, Any]:
        """
        Обучает модель
        
        Args:
            data_yaml: путь к data.yaml
            epochs: количество эпох
            imgsz: размер изображения
            batch: размер батча
            lr0: начальный learning rate
            lrf: финальный learning rate
            cos_lr: использовать cosine annealing
            freeze: количество замороженных слоев
            mixup: коэффициент mixup аугментации
            workers: количество воркеров
            cache: кэшировать данные
            **kwargs: дополнительные параметры
            
        Returns:
            словарь с результатами обучения
        """
        if self.model is None:
            self.load_model()
        
        # Проверяем существование data.yaml
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"data.yaml не найден: {data_yaml}")
        
        # Формируем название эксперимента
        exp_name = f"yolo_{epochs}e_b{batch}_f{freeze}_m{mixup}"
        
        print("\n" + "="*50)
        print(f"🚀 НАЧАЛО ОБУЧЕНИЯ YOLO-World")
        print("="*50)
        print(f"📊 Параметры:")
        print(f"  data:    {data_yaml}")
        print(f"  epochs:  {epochs}")
        print(f"  imgsz:   {imgsz}")
        print(f"  batch:   {batch}")
        print(f"  lr0:     {lr0}")
        print(f"  freeze:  {freeze}")
        print(f"  mixup:   {mixup}")
        print(f"  exp:     {exp_name}")
        print("="*50 + "\n")
        
        # Засекаем время
        timer = Timer("Обучение")
        
        with timer:
            # Обучение
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                lrf=lrf,
                cos_lr=cos_lr,
                freeze=freeze,
                mixup=mixup,
                workers=workers,
                cache=cache,
                **kwargs
            )
        
        timer.print_elapsed()
        
        # Находим путь к лучшим весам
        best_weights_path = Path("runs/detect/train/weights/best.pt")
        if not best_weights_path.exists():
            # Пробуем другие возможные пути
            possible_paths = [
                Path("runs/detect/train/weights/best.pt"),
                Path("runs/detect/train2/weights/best.pt"),
                Path("runs/detect/train3/weights/best.pt")
            ]
            for path in possible_paths:
                if path.exists():
                    best_weights_path = path
                    break
        
        result = {
            "experiment_name": exp_name,
            "best_weights_path": str(best_weights_path) if best_weights_path.exists() else None,
            "training_time_seconds": timer.elapsed,
            "training_time_formatted": timer.get_elapsed_formatted()
        }
        
        return result
    
    def validate(self,
                data_yaml: str,
                split: str = "val",
                imgsz: int = 640,
                batch: int = 16,
                weights_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Валидация модели
        
        Args:
            data_yaml: путь к data.yaml
            split: сплит для валидации ('val', 'test')
            imgsz: размер изображения
            batch: размер батча
            weights_path: путь к весам (если None, используется последняя обученная модель)
            
        Returns:
            словарь с метриками
        """
        if self.model is None:
            self.load_model()
        
        print(f"\n🔍 Валидация модели на {split} сплите...")
        
        # Если указан путь к весам, загружаем их
        if weights_path and os.path.exists(weights_path):
            print(f"📥 Загрузка весов из {weights_path}")
            self.model = YOLOWorld(weights_path)
        
        # Валидация
        results = self.model.val(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            batch=batch
        )
        
        # Определяем название эксперимента
        if weights_path:
            exp_name = Path(weights_path).stem
        else:
            exp_name = "latest_model"
        
        # Извлекаем метрики
        metrics = self.metrics_calculator.extract_from_yolo_results(results, exp_name)
        
        # Сохраняем метрики
        metrics_path = self.metrics_calculator.save_metrics(metrics)
        
        # Печатаем сводку
        self.metrics_calculator.print_summary(metrics)
        
        return {
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "results": results
        }
    
    def train_and_validate(self, 
                          data_yaml: str,
                          epochs: int = 3,
                          **kwargs) -> Dict[str, Any]:
        """
        Обучает модель и сразу проводит валидацию
        
        Returns:
            словарь с результатами обучения и метриками
        """
        # Обучаем
        train_result = self.train(data_yaml, epochs=epochs, **kwargs)
        
        # Валидируем
        val_result = self.validate(
            data_yaml=data_yaml,
            split="val",
            weights_path=train_result["best_weights_path"]
        )
        
        return {
            "train": train_result,
            "validation": val_result
        }