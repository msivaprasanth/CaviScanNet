"""Utility functions for dental caries detection and classification."""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

def allowed_file(filename, allowed_extensions):
    """Check if a filename has an allowed extension.
    
    Args:
        filename (str): Name of the file
        allowed_extensions (set): Set of allowed file extensions
        
    Returns:
        bool: True if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def prepare_image(image, size):
    """Prepare image for model input.
    
    Args:
        image (PIL.Image): Input image
        size (tuple): Target size (height, width)
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def draw_detections(image, detections_by_severity, colors):
    """Draw bounding boxes and labels on image.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        detections_by_severity (dict): Dictionary of detections organized by severity
        colors (dict): Dictionary mapping severity to BGR colors
        
    Returns:
        numpy.ndarray: Image with drawn detections
    """
    result = image.copy()
    
    for severity, detections in detections_by_severity.items():
        color = colors[severity]
        for detection in detections:
            x1, y1, x2, y2 = detection['coords']
            score = detection['score']
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{severity}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            y1_label = max(y1, label_size[1])
            cv2.rectangle(result, (x1, y1_label - label_size[1]),
                        (x1 + label_size[0], y1_label + 5), color, -1)
            cv2.putText(result, label, (x1, y1_label),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def ensure_rgb(image):
    """Ensure image is in RGB format.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: RGB image
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def normalize_box_coordinates(boxes, width, height):
    """Normalize box coordinates to [0, 1] range.
    
    Args:
        boxes (numpy.ndarray): Array of box coordinates [x1, y1, x2, y2]
        width (int): Image width
        height (int): Image height
        
    Returns:
        numpy.ndarray: Normalized box coordinates
    """
    boxes = boxes.copy()
    boxes[:, [0, 2]] /= width
    boxes[:, [1, 3]] /= height
    return boxes

def clip_boxes(boxes, width, height):
    """Clip box coordinates to image boundaries.
    
    Args:
        boxes (numpy.ndarray): Array of box coordinates [x1, y1, x2, y2]
        width (int): Image width
        height (int): Image height
        
    Returns:
        numpy.ndarray: Clipped box coordinates
    """
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes

def is_edge_box(box, width, height, edge_threshold=0.03):
    """Check if bounding box is near image edges.
    
    Args:
        box (list): Bounding box coordinates [x1, y1, x2, y2]
        width (int): Image width
        height (int): Image height
        edge_threshold (float): Threshold for edge detection as percentage of image dimension
        
    Returns:
        bool: True if box is near edge, False otherwise
    """
    x1, y1, x2, y2 = box
    
    # Calculate edge boundaries
    edge_x = width * edge_threshold
    edge_y = height * edge_threshold
    
    # Check if box touches edges
    return (x1 < edge_x or  # Left edge
            y1 < edge_y or  # Top edge
            x2 > width - edge_x or  # Right edge
            y2 > height - edge_y)  # Bottom edge 