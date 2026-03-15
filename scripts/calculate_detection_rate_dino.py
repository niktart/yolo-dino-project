#!/usr/bin/env python
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.grounding_dino import GroundingDINOModel
from src.evaluation.detection_rate import DetectionRateCalculator
from config.settings import GROUNDING_DINO_CONFIG, DETECTION_RATE_CONFIG

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True,
                       help='Путь к обученным весам')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--text_prompt', type=str, required=True)
    parser.add_argument('--max_images', type=int, 
                       default=DETECTION_RATE_CONFIG['max_images'])
    parser.add_argument('--conf_threshold', type=float,
                       default=DETECTION_RATE_CONFIG['conf_threshold'])
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Загружаем модель
    model = GroundingDINOModel(
        config_path=GROUNDING_DINO_CONFIG['config_path'],
        weights_path=GROUNDING_DINO_CONFIG['pretrained_weights']
    )
    model.load_pretrained()
    model.load_trained(args.weights)
    
    # Создаем калькулятор
    calculator = DetectionRateCalculator(model, args.text_prompt)
    
    # Считаем detection rate
    rate, df = calculator.calculate(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        max_images=args.max_images,
        conf_threshold=args.conf_threshold,
        text_threshold=DETECTION_RATE_CONFIG['text_threshold'],
        iou_threshold=DETECTION_RATE_CONFIG['iou_threshold']
    )
    
    # Сохраняем результаты
    df.to_csv("detection_results.csv", index=False)
    print(f"\n💾 Результаты сохранены в detection_results.csv")

if __name__ == "__main__":
    main()