# src/data/adapters/csv_dataset.py

import pandas as pd
import os


class CSVDatasetAdapter:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def get_images(self, images_dir):
        return sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

    def get_gt(self, image_path):
        img_name = os.path.basename(image_path)

        df_img = self.df[self.df['image_name'] == img_name]

        boxes = []
        for _, row in df_img.iterrows():
            x, y, w, h = row[['bbox_x','bbox_y','bbox_width','bbox_height']]
            boxes.append([x, y, x+w, y+h])

        return boxes
