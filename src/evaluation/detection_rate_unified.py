# src/evaluation/detection_rate_unified.py

import os
import cv2
import pandas as pd
from tqdm import tqdm


class DetectionRateCalculator:
    def __init__(self, model_adapter, dataset_adapter):
        self.model = model_adapter
        self.dataset = dataset_adapter

    @staticmethod
    def calculate_iou(box1, box2):
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

    def _match_boxes(self, gt_boxes, pred_boxes, iou_threshold):
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

    def calculate(self, images_dir, iou_threshold=0.5, conf_threshold=0.3):
        image_files = self.dataset.get_images(images_dir)

        total = 0
        detected = 0
        results = []

        for img_file in tqdm(image_files, desc="Detection rate"):
            image_path = os.path.join(images_dir, img_file)

            gt_boxes = self.dataset.get_gt(image_path)
            if len(gt_boxes) == 0:
                continue

            pred_boxes = self.model.predict(image_path, conf_threshold)

            n_gt = len(gt_boxes)
            total += 1

            matched_gt, _ = self._match_boxes(
                gt_boxes, pred_boxes, iou_threshold
            )

            is_detected = (len(matched_gt) == n_gt) and (len(pred_boxes) == n_gt)

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

        print(f"\n🎯 DETECTION RATE: {rate:.2%}")
        print(f"Total: {total}, Detected: {detected}")

        return rate, pd.DataFrame(results)
