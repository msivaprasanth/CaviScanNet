"""Training script for dental caries classification model."""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from model import DentalCariesClassifier
from dataset import DentalDataset

def train_model(args):
    """Train the classification model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data augmentation and preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DentalDataset(
        root_dir=args.data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = DentalDataset(
        root_dir=args.data_dir,
        split='val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = DentalCariesClassifier()
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1
    )
    
    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training phase
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        
        # Log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        print(f'\nEpoch {epoch+1}/{args.num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print('Saved best model checkpoint')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
    
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train dental caries classification model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, required=True,
                      help='Directory to save tensorboard logs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
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