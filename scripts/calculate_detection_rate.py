# scripts/calculate_detection_rate.py

import argparse
from ultralytics import YOLOWorld

from src.evaluation.detection_rate_unified import DetectionRateCalculator
from src.models.adapters.yolo_adapter import YOLOAdapter
from src.models.adapters.dino_adapter import GroundingDINOAdapter
from src.data.adapters.yolo_dataset import YOLODatasetAdapter
from src.data.adapters.csv_dataset import CSVDatasetAdapter
from src.models.grounding_dino import GroundingDINOModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["yolo", "dino"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images_dir", required=True)

    parser.add_argument("--labels_dir")
    parser.add_argument("--labels_csv")
    parser.add_argument("--text_prompt")

    args = parser.parse_args()

    if args.model == "yolo":
        model = YOLOWorld(args.weights)
        model_adapter = YOLOAdapter(model)
        dataset_adapter = YOLODatasetAdapter(args.labels_dir)

    elif args.model == "dino":
        dino = GroundingDINOModel("config.py", args.weights)
        dino.load_pretrained()
        dino.load_trained(args.weights)

        model_adapter = GroundingDINOAdapter(
            dino,
            args.text_prompt
        )
        dataset_adapter = CSVDatasetAdapter(args.labels_csv)

    calculator = DetectionRateCalculator(model_adapter, dataset_adapter)

    calculator.calculate(args.images_dir)


if __name__ == "__main__":
    main()
