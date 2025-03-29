"""Dataset class for dental caries detection."""

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class DentalDetectionDataset(Dataset):
    """Dataset class for dental X-ray images object detection."""
    
    def __init__(self, root_dir, split='train', transform=False):
        """Initialize the dataset.
        
        Args:
            root_dir (str): Root directory containing the dataset
            split (str): Dataset split ('train', 'val', or 'test')
            transform (bool): Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.split = split
        
        # Load annotations
        annotations_file = os.path.join(root_dir, f'{split}_annotations.json')
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Setup transforms
        if transform:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                 rotate_limit=10, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc',
                                      label_fields=['labels']))
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc',
                                      label_fields=['labels']))
    
    def __len__(self):
        """Return the size of dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get item by index.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where target is a dictionary containing:
                  - boxes (FloatTensor[N, 4]): bounding boxes in [x1, y1, x2, y2] format
                  - labels (Int64Tensor[N]): the label for each bounding box
                  - image_id (Int64Tensor[1]): image identifier
                  - area (Tensor[N]): area of the bounding boxes
                  - iscrowd (UInt8Tensor[N]): instances are crowd
        """
        # Load image and annotations
        img_info = self.annotations[idx]
        img_path = os.path.join(self.root_dir, 'images', img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Convert PIL image to numpy array
        image = np.array(image)
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        for ann in img_info['annotations']:
            boxes.append(ann['bbox'])  # [x1, y1, x2, y2]
            labels.append(ann['category_id'])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        
        # Prepare target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.uint8)
        
        return image, target

def collate_fn(batch):
    """Custom collate function for DataLoader.
    
    Args:
        batch: List of tuples (image, target)
        
    Returns:
        tuple: (images, targets) where:
              - images is a list of tensors
              - targets is a list of dictionaries
    """
    return tuple(zip(*batch)) 