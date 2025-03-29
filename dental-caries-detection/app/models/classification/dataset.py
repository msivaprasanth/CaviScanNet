"""Dataset class for dental caries classification."""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

class DentalDataset(Dataset):
    """Dataset class for dental X-ray images classification."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """Initialize the dataset.
        
        Args:
            root_dir (str): Root directory containing the dataset
            split (str): Dataset split ('train', 'val', or 'test')
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load annotations
        annotations_file = os.path.join(root_dir, f'{split}_annotations.csv')
        self.annotations = pd.read_csv(annotations_file)
        
        # Class mapping
        self.class_to_idx = {
            'normal': 0,
            'superficial': 1,
            'medium': 2,
            'deep': 3
        }
    
    def __len__(self):
        """Return the size of dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get item by index.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, label) where label is index of the target class
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        img_name = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        label = self.class_to_idx[self.annotations.iloc[idx, 1]]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label 