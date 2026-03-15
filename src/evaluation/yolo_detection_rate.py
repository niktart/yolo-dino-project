import os
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional, Any


class YOLODetectionRateCalculator:
    """
    Класс для расчета доли изображений, распознанных на 100%
    
    Фото считается распознанным только если:
        1) Количество предсказанных bbox == количеству GT
        2) Все GT bbox распознаны (100% recall)
    """
    
    def __init__(self, model, device: str = None):
        """
        Args:
            model: загруженная YOLO модель
            device: устройство для вычислений
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    @staticmethod
    def load_label(label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загрузка GT аннотаций в формате YOLO
        
        Args:
            label_path: путь к файлу с аннотацией
            
        Returns:
            (boxes, classes) - массивы с bbox и классами
        """
        gt_boxes = []
        gt_classes = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        gt_boxes.append([x_center, y_center, width, height])
                        gt_classes.append(class_id)

        return np.array(gt_boxes), np.array(gt_classes)
    
    @staticmethod
    def yolo_to_bbox(yolo_box: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Конвертирует YOLO bbox (нормализованный) в абсолютные координаты xyxy
        
        Args:
            yolo_box: [x_center, y_center, width, height] (нормализовано)
            img_width: ширина изображения в пикселях
            img_height: высота изображения в пикселях
            
        Returns:
            [x1, y1, x2, y2] - абсолютные координаты
        """
        x_center, y_center, width, height = yolo_box
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        x2 = (x_center + width/2) * img_width
        y2 = (y_center + height/2) * img_height
        return [x1, y1, x2, y2]
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Расчет IoU между двумя bbox
        
        Args:
            box1, box2: [x1, y1, x2, y2]
            
        Returns:
            IoU значение
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
    
    def match_predictions_to_gt(self, 
                                gt_boxes: List[List[float]], 
                                pred_boxes: List[List[float]], 
                                iou_threshold: float = 0.5) -> Tuple[set, set]:
        """
        Жадное сопоставление предсказанных bbox с GT
        
        Args:
            gt_boxes: список GT bbox
            pred_boxes: список предсказанных bbox
            iou_threshold: порог IoU для сопоставления
            
        Returns:
            (matched_gt_indices, matched_pred_indices) - множества индексов
        """
        matched_gt = set()
        matched_pred = set()

        for i, gt in enumerate(gt_boxes):
            best_iou = 0
            best_j = -1

            for j, pred in enumerate(pred_boxes):
                if j in matched_pred:
                    continue

                iou = self.calculate_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold:
                matched_gt.add(i)
                matched_pred.add(best_j)
        
        return matched_gt, matched_pred
    
    def process_single_image(self, 
                            image_path: str, 
                            label_path: str,
                            conf_threshold: float = 0.15,
                            iou_threshold: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Обрабатывает одно изображение и возвращает результаты
        
        Args:
            image_path: путь к изображению
            label_path: путь к файлу с аннотацией
            conf_threshold: порог уверенности для детекции
            iou_threshold: порог IoU для сопоставления
            
        Returns:
            словарь с результатами или None если изображение не валидно
        """
        # Проверяем существование label
        if not os.path.exists(label_path):
            return None

        # Загружаем GT
        gt_boxes_yolo, _ = self.load_label(label_path)
        if len(gt_boxes_yolo) == 0:
            return None

        # Загружаем изображение для получения размеров
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        
        # Конвертируем GT bbox в абсолютные координаты
        gt_boxes = [self.yolo_to_bbox(box, w, h) for box in gt_boxes_yolo]
        n_gt = len(gt_boxes)

        # Инференс
        with torch.no_grad():
            preds = self.model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]

        # Извлекаем предсказанные bbox
        pred_boxes = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for box in preds.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2, y2])

        n_pred = len(pred_boxes)

        # Сопоставляем предсказания с GT
        matched_gt, matched_pred = self.match_predictions_to_gt(
            gt_boxes, pred_boxes, iou_threshold
        )

        # Проверяем условия 100% распознавания
        all_gt_matched = (len(matched_gt) == n_gt)
        same_number_boxes = (n_pred == n_gt)
        is_detected = all_gt_matched and same_number_boxes

        return {
            "image": os.path.basename(image_path),
            "gt_objects": n_gt,
            "pred_objects": n_pred,
            "matched": len(matched_gt),
            "all_gt_matched": all_gt_matched,
            "same_number_boxes": same_number_boxes,
            "detected": is_detected,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes
        }
    
    def calculate_detection_rate(self,
                                images_dir: str,
                                labels_dir: str,
                                conf_threshold: float = 0.15,
                                iou_threshold: float = 0.5,
                                save_results: bool = True,
                                output_csv: Optional[str] = None) -> Tuple[float, pd.DataFrame]:
        """
        Рассчитывает долю полностью распознанных изображений
        
        Args:
            images_dir: директория с изображениями
            labels_dir: директория с аннотациями
            conf_threshold: порог уверенности
            iou_threshold: порог IoU
            save_results: сохранять ли результаты в CSV
            output_csv: путь для сохранения CSV (если None, генерируется автоматически)
            
        Returns:
            (detection_rate, DataFrame с результатами)
        """
        # Получаем список изображений
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        print(f"📸 Найдено {len(image_files)} изображений")
        print(f"📊 Параметры:")
        print(f"  conf_threshold = {conf_threshold}")
        print(f"  iou_threshold = {iou_threshold}")
        print("="*50)

        total_images = 0
        detected_images = 0
        results = []

        # Обрабатываем каждое изображение
        for img_file in tqdm(image_files, desc="Расчет detection rate"):
            image_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(
                labels_dir,
                Path(img_file).with_suffix(".txt").name
            )

            result = self.process_single_image(
                image_path, 
                label_path,
                conf_threshold,
                iou_threshold
            )

            if result is not None:
                total_images += 1
                if result["detected"]:
                    detected_images += 1
                results.append(result)

        # Рассчитываем итоговую долю
        detection_rate = detected_images / total_images if total_images > 0 else 0

        # Создаем DataFrame
        df = pd.DataFrame(results)

        # Выводим результаты
        self._print_summary(detection_rate, total_images, detected_images)

        # Сохраняем в CSV
        if save_results:
            if output_csv is None:
                output_csv = f"detection_rate_conf{conf_threshold}_iou{iou_threshold}.csv"
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Результаты сохранены в {output_csv}")

        return detection_rate, df
    
    def _print_summary(self, detection_rate: float, total: int, detected: int):
        """Печатает сводку результатов"""
        print("\n" + "="*50)
        print(f"🎯 STRICT DETECTION RATE (100% only): {detection_rate:.2%}")
        print("="*50)
        print(f"📊 Статистика:")
        print(f"  Всего изображений с GT: {total}")
        print(f"  Полностью распознано: {detected}")
        print(f"  Частично/не распознано: {total - detected}")
        print(f"  Доля: {detection_rate:.2%}")
        
        # Дополнительная статистика
        if total > 0:
            avg_gt = np.mean([r["gt_objects"] for r in results]) if 'results' in locals() else 0
            avg_pred = np.mean([r["pred_objects"] for r in results]) if 'results' in locals() else 0
            print(f"\n📈 Среднее количество объектов:")
            print(f"  GT: {avg_gt:.2f}")
            print(f"  Pred: {avg_pred:.2f}")
    
    def analyze_errors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализирует ошибки: почему изображения не были распознаны
        
        Args:
            df: DataFrame с результатами из calculate_detection_rate
            
        Returns:
            словарь со статистикой ошибок
        """
        if df.empty:
            return {}
        
        missed = df[~df["detected"]]
        
        # Категории ошибок
        wrong_count = missed[~missed["same_number_boxes"]]
        not_all_matched = missed[~missed["all_gt_matched"]]
        
        # Пересечение категорий
        both_errors = missed[
            (~missed["same_number_boxes"]) & (~missed["all_gt_matched"])
        ]
        
        stats = {
            "total_missed": len(missed),
            "wrong_object_count": len(wrong_count),
            "not_all_matched": len(not_all_matched),
            "both_errors": len(both_errors),
            "avg_gt_missed": missed["gt_objects"].mean(),
            "avg_pred_missed": missed["pred_objects"].mean(),
            "avg_matched_missed": missed["matched"].mean()
        }
        
        print("\n" + "="*50)
        print("🔍 АНАЛИЗ ОШИБОК")
        print("="*50)
        print(f"Всего пропущено изображений: {stats['total_missed']}")
        print(f"  ❌ Неверное количество объектов: {stats['wrong_object_count']}")
        print(f"  ❌ Не все GT сопоставлены: {stats['not_all_matched']}")
        print(f"  ❌ Обе ошибки: {stats['both_errors']}")
        
        return stats
    
    def calculate_by_confidence_threshold(self,
                                        images_dir: str,
                                        labels_dir: str,
                                        thresholds: List[float] = [0.1, 0.15, 0.2, 0.25, 0.3],
                                        iou_threshold: float = 0.5) -> pd.DataFrame:
        """
        Рассчитывает detection rate для разных порогов уверенности
        
        Args:
            images_dir: директория с изображениями
            labels_dir: директория с аннотациями
            thresholds: список порогов уверенности
            iou_threshold: порог IoU
            
        Returns:
            DataFrame с результатами для каждого порога
        """
        results = []
        
        for conf in thresholds:
            print(f"\n📊 Расчет для conf_threshold = {conf}")
            rate, _ = self.calculate_detection_rate(
                images_dir=images_dir,
                labels_dir=labels_dir,
                conf_threshold=conf,
                iou_threshold=iou_threshold,
                save_results=False
            )
            results.append({
                "conf_threshold": conf,
                "detection_rate": rate
            })
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*50)
        print("📊 DETECTION RATE ПО ПОРОГАМ")
        print("="*50)
        for _, row in df.iterrows():
            print(f"  conf={row['conf_threshold']}: {row['detection_rate']:.2%}")
        
        return df