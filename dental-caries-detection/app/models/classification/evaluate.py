"""Evaluation script for dental caries classification model."""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import DentalCariesClassifier
from dataset import DentalDataset
from torchvision import transforms

def evaluate_model(args):
    """Evaluate the classification model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preprocessing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    test_dataset = DentalDataset(
        root_dir=args.data_dir,
        split='test',
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load model
    model = DentalCariesClassifier()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()
    
    # Lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    
    # Evaluate model
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    class_names = ['normal', 'superficial', 'medium', 'deep']
    report = classification_report(all_labels, all_preds,
                                 target_names=class_names,
                                 digits=3)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print classification report
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save confusion matrix plot
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    print(f"\nConfusion matrix saved to {args.output_dir}/confusion_matrix.png")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate dental caries classification model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluate_model(args)

if __name__ == '__main__':
    main() 