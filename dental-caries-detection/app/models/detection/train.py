"""Training script for dental caries detection model."""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm

from model import DentalCariesDetector
from dataset import DentalDetectionDataset
from utils import collate_fn

def train_model(args):
    """Train the detection model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = DentalDetectionDataset(
        root_dir=args.data_dir,
        split='train',
        transform=True
    )
    
    val_dataset = DentalDetectionDataset(
        root_dir=args.data_dir,
        split='val',
        transform=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = DentalCariesDetector()
    model = model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                        step_size=3,
                                        gamma=0.1)
    
    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        # Training phase
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for images, targets in train_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            train_bar.set_postfix({'loss': f'{losses.item():.4f}'})
        
        # Calculate average training loss
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f'\nEpoch {epoch+1}/{args.num_epochs}:')
        print(f'Train Loss: {avg_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(args.model_dir, 'best_model.pth'))
            print('Saved best model checkpoint')
        
        # Learning rate scheduling
        scheduler.step()
    
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train dental caries detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, required=True,
                      help='Directory to save tensorboard logs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=25,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                      help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    train_model(args)

if __name__ == '__main__':
    main() 