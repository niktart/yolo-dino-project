# src/models/adapters/dino_adapter.py

import torch
import cv2


class GroundingDINOAdapter:
    def __init__(self, model, text_prompt, text_threshold=0.25):
        self.model = model
        self.text_prompt = text_prompt
        self.text_threshold = text_threshold

    def predict(self, image_path, conf_threshold):
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        _, boxes, _, _ = self.model.predict_image(
            image_path,
            self.text_prompt,
            conf_threshold,
            self.text_threshold
        )

        pred_boxes = []

        if len(boxes) > 0:
            boxes = boxes * torch.tensor([w, h, w, h])
            for cx, cy, bw, bh in boxes:
                x1 = cx - bw/2
                y1 = cy - bh/2
                x2 = cx + bw/2
                y2 = cy + bh/2
                pred_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])

        return pred_boxes
