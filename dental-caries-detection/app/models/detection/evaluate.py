"""Evaluation script for dental caries detection model."""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

from model import DentalCariesDetector
from dataset import DentalDetectionDataset, collate_fn
from utils import draw_boxes

def evaluate_model(args):
    """Evaluate the detection model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    test_dataset = DentalDetectionDataset(
        root_dir=args.data_dir,
        split='test',
        transform=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Load model
    model = DentalCariesDetector()
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Lists to store predictions and ground truth
    predictions = []
    ground_truth = []
    image_ids = []
    
    # Evaluate model
    print("\nRunning evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = list(img.to(device) for img in images)
            
            # Get predictions
            outputs = model(images)
            
            # Process each image in the batch
            for i, (output, target) in enumerate(zip(outputs, targets)):
                image_id = target['image_id'].item()
                image_ids.append(image_id)
                
                # Process predictions
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Store predictions in COCO format
                for box, score, label in zip(boxes, scores, labels):
                    if score > args.score_threshold:
                        predictions.append({
                            'image_id': image_id,
                            'category_id': label,
                            'bbox': box.tolist(),
                            'score': float(score)
                        })
                
                # Store ground truth in COCO format
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                for box, label in zip(gt_boxes, gt_labels):
                    ground_truth.append({
                        'image_id': image_id,
                        'category_id': label,
                        'bbox': box.tolist(),
                        'area': (box[2] - box[0]) * (box[3] - box[1]),
                        'iscrowd': 0
                    })
    
    # Save predictions and ground truth
    pred_file = os.path.join(args.output_dir, 'predictions.json')
    gt_file = os.path.join(args.output_dir, 'ground_truth.json')
    
    # Create COCO format annotations
    gt_dict = {
        'images': [{'id': id} for id in set(image_ids)],
        'categories': [{'id': i} for i in range(1, 4)],  # 3 classes
        'annotations': ground_truth
    }
    
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)
    
    with open(gt_file, 'w') as f:
        json.dump(gt_dict, f)
    
    # Evaluate using COCO metrics
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Save evaluation results
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("COCO Evaluation Results:\n\n")
        f.write(f"Average Precision (AP) @ IoU=0.50:0.95: {coco_eval.stats[0]:.3f}\n")
        f.write(f"Average Precision (AP) @ IoU=0.50: {coco_eval.stats[1]:.3f}\n")
        f.write(f"Average Precision (AP) @ IoU=0.75: {coco_eval.stats[2]:.3f}\n")
        f.write(f"Average Precision (AP) small: {coco_eval.stats[3]:.3f}\n")
        f.write(f"Average Precision (AP) medium: {coco_eval.stats[4]:.3f}\n")
        f.write(f"Average Precision (AP) large: {coco_eval.stats[5]:.3f}\n")
        f.write(f"Average Recall (AR) @ IoU=0.50:0.95: {coco_eval.stats[6]:.3f}\n")
    
    print(f"\nEvaluation results saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate dental caries detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                      help='Score threshold for predictions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluate_model(args)

if __name__ == '__main__':
    main() 