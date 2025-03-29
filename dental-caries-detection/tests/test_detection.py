"""Test cases for dental caries detection model."""

import os
import pytest
import torch
import numpy as np
import json
from PIL import Image

from app.models.detection.model import DentalCariesDetector
from app.models.detection.dataset import DentalDetectionDataset, collate_fn

@pytest.fixture
def model():
    """Create model instance for testing."""
    return DentalCariesDetector()

@pytest.fixture
def sample_target():
    """Create sample target for testing."""
    return {
        'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'image_id': torch.tensor([0]),
        'area': torch.tensor([10000.], dtype=torch.float32),
        'iscrowd': torch.tensor([0], dtype=torch.uint8)
    }

def test_model_output_format(model):
    """Test if model outputs correct format."""
    # Create dummy input
    image = torch.randn(3, 800, 800)
    images = [image]
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    assert isinstance(outputs, list), "Output should be a list"
    assert len(outputs) == 1, "Output list should have length 1"
    
    output = outputs[0]
    assert 'boxes' in output, "Output should contain 'boxes'"
    assert 'labels' in output, "Output should contain 'labels'"
    assert 'scores' in output, "Output should contain 'scores'"

def test_model_training_mode(model, sample_target):
    """Test model in training mode."""
    # Create dummy input
    image = torch.randn(3, 800, 800)
    images = [image]
    targets = [sample_target]
    
    # Run training step
    model.train()
    loss_dict = model(images, targets)
    
    assert isinstance(loss_dict, dict), "Training should return loss dictionary"
    assert 'loss_classifier' in loss_dict, "Loss dict should contain classifier loss"
    assert 'loss_box_reg' in loss_dict, "Loss dict should contain box regression loss"
    assert 'loss_objectness' in loss_dict, "Loss dict should contain objectness loss"
    assert 'loss_rpn_box_reg' in loss_dict, "Loss dict should contain RPN box regression loss"

def test_dataset_loading():
    """Test detection dataset loading and item retrieval."""
    # Create temporary dataset structure
    root_dir = "test_data"
    os.makedirs(os.path.join(root_dir, "images"), exist_ok=True)
    
    # Create dummy image
    image = Image.new('RGB', (800, 800))
    image.save(os.path.join(root_dir, "images", "test.png"))
    
    # Create annotations file
    annotations = {
        "images": [
            {
                "file_name": "test.png",
                "annotations": [
                    {
                        "bbox": [100, 100, 200, 200],
                        "category_id": 1
                    }
                ]
            }
        ]
    }
    
    with open(os.path.join(root_dir, "train_annotations.json"), "w") as f:
        json.dump(annotations, f)
    
    # Create dataset
    dataset = DentalDetectionDataset(root_dir, "train")
    
    assert len(dataset) == 1, f"Expected dataset length 1, got {len(dataset)}"
    
    # Test item retrieval
    image, target = dataset[0]
    assert isinstance(image, torch.Tensor), "Dataset should return tensor"
    assert isinstance(target, dict), "Target should be dictionary"
    assert 'boxes' in target, "Target should contain boxes"
    assert 'labels' in target, "Target should contain labels"
    
    # Cleanup
    import shutil
    shutil.rmtree(root_dir)

def test_model_prediction_boxes(model):
    """Test if model predictions have valid box coordinates."""
    # Create dummy input
    image = torch.randn(3, 800, 800)
    images = [image]
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    boxes = outputs[0]['boxes']
    assert torch.all(boxes[:, 0] <= boxes[:, 2]), "x1 should be <= x2"
    assert torch.all(boxes[:, 1] <= boxes[:, 3]), "y1 should be <= y2"
    assert torch.all(boxes >= 0), "Box coordinates should be non-negative"

def test_collate_fn():
    """Test collate function for data loader."""
    # Create dummy batch
    image1 = torch.randn(3, 800, 800)
    image2 = torch.randn(3, 800, 800)
    target1 = {
        'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64)
    }
    target2 = {
        'boxes': torch.tensor([[300, 300, 400, 400]], dtype=torch.float32),
        'labels': torch.tensor([2], dtype=torch.int64)
    }
    
    batch = [(image1, target1), (image2, target2)]
    images, targets = collate_fn(batch)
    
    assert len(images) == 2, "Should have 2 images"
    assert len(targets) == 2, "Should have 2 targets"
    assert torch.equal(images[0], image1), "First image should match"
    assert torch.equal(targets[0]['boxes'], target1['boxes']), "First target boxes should match" 