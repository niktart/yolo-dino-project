# src/data/adapters/yolo_dataset.py

import os
import cv2
from pathlib import Path


class YOLODatasetAdapter:
    def __init__(self, labels_dir):
        self.labels_dir = labels_dir

    def get_images(self, images_dir):
        return sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

    def get_gt(self, image_path):
        label_path = os.path.join(
            self.labels_dir,
            Path(image_path).with_suffix(".txt").name
        )

        if not os.path.exists(label_path):
            return []

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        boxes = []
        with open(label_path) as f:
            for line in f:
                _, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                boxes.append([x1, y1, x2, y2])

        return boxes
