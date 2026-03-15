import torch
import cv2
import pandas as pd
from tqdm import tqdm
from groundingdino.util.inference import load_image, predict

class DetectionRateCalculator:
    def __init__(self, model, text_prompt):
        self.model = model
        self.text_prompt = text_prompt
        
    @staticmethod
    def calculate_iou(box1, box2):
        """IoU двух bbox [x1,y1,x2,y2]"""
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
    
    def calculate(self, images_dir, labels_csv, max_images=100,
                  conf_threshold=0.3, text_threshold=0.25, iou_threshold=0.5):
        """Расчет detection rate"""
        
        # Читаем CSV
        df_csv = pd.read_csv(labels_csv)
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:max_images]
        
        total = 0
        detected = 0
        results = []
        
        for img_file in tqdm(image_files, desc="Detection rate"):
            image_path = os.path.join(images_dir, img_file)
            
            # GT из CSV
            df_img = df_csv[df_csv['image_name'] == img_file]
            if df_img.empty:
                continue
                
            gt_boxes_raw = df_img[['bbox_x','bbox_y','bbox_width','bbox_height']].values
            gt_boxes = [[x, y, x+bw, y+bh] for x, y, bw, bh in gt_boxes_raw]
            
            if len(gt_boxes) == 0:
                continue
                
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            n_gt = len(gt_boxes)
            total += 1
            
            # Инференс
            _, image_tensor = load_image(image_path)
            with torch.no_grad():
                boxes, _, _ = self.model.predict_image(
                    image_path, self.text_prompt,
                    conf_threshold, text_threshold
                )
            
            # Конвертируем предсказания
            pred_boxes = []
            if len(boxes) > 0:
                boxes = boxes * torch.tensor([w, h, w, h])
                for cx, cy, bw, bh in boxes:
                    x1 = cx - bw/2
                    y1 = cy - bh/2
                    x2 = cx + bw/2
                    y2 = cy + bh/2
                    pred_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
            
            # Matching
            matched_gt, matched_pred = self._match_boxes(
                gt_boxes, pred_boxes, iou_threshold
            )
            
            # Строгая проверка
            all_gt_matched = (len(matched_gt) == n_gt)
            same_number = (len(pred_boxes) == n_gt)
            is_detected = all_gt_matched and same_number
            
            if is_detected:
                detected += 1
                
            results.append({
                "image": img_file,
                "gt_objects": n_gt,
                "pred_objects": len(pred_boxes),
                "matched": len(matched_gt),
                "detected": is_detected
            })
        
        rate = detected / total if total > 0 else 0
        
        self._print_results(rate, total, detected)
        
        return rate, pd.DataFrame(results)
    
    def _match_boxes(self, gt_boxes, pred_boxes, iou_threshold):
        """Жадное сопоставление боксов"""
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
    
    def _print_results(self, rate, total, detected):
        print("\n" + "="*50)
        print(f"🎯 GROUNDING DINO DETECTION RATE: {rate:.2%}")
        print("="*50)
        print(f"Total: {total}")
        print(f"Detected: {detected}")
        print(f"Missed: {total - detected}")