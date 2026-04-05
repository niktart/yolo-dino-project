# src/models/adapters/yolo_adapter.py

import torch


class YOLOAdapter:
    def __init__(self, model):
        self.model = model

    def predict(self, image_path, conf_threshold):
        with torch.no_grad():
            preds = self.model.predict(
                image_path,
                conf=conf_threshold,
                verbose=False
            )[0]

        boxes = []
        if preds.boxes is not None:
            for box in preds.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])

        return boxes
