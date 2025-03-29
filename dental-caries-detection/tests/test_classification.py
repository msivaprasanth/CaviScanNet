"""Test cases for dental caries classification model."""

import os
import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from app.models.classification.model import DentalCariesClassifier
from app.models.classification.dataset import DentalDataset

@pytest.fixture
def model():
    """Create model instance for testing."""
    return DentalCariesClassifier()

@pytest.fixture
def transform():
    """Create transform for testing."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def test_model_output_shape(model):
    """Test if model outputs correct shape."""
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 4), \
        f"Expected output shape (4, 4), got {output.shape}"

def test_model_forward_pass(model, transform):
    """Test model forward pass with transformed image."""
    # Create dummy image
    image = Image.new('RGB', (300, 300))
    input_tensor = transform(image).unsqueeze(0)
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == (1, 4), \
        f"Expected output shape (1, 4), got {output.shape}"
    assert torch.is_tensor(output), "Output should be a tensor"

def test_model_prediction(model, transform):
    """Test model prediction."""
    # Create dummy image
    image = Image.new('RGB', (300, 300))
    input_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    assert prediction.item() in [0, 1, 2, 3], \
        f"Prediction {prediction.item()} not in valid range [0, 1, 2, 3]"
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0)), \
        "Probabilities should sum to 1"

def test_dataset_loading():
    """Test dataset loading and item retrieval."""
    # Create temporary dataset structure
    root_dir = "test_data"
    os.makedirs(os.path.join(root_dir, "images"), exist_ok=True)
    
    # Create dummy image
    image = Image.new('RGB', (300, 300))
    image.save(os.path.join(root_dir, "images", "test.png"))
    
    # Create annotations file
    with open(os.path.join(root_dir, "train_annotations.csv"), "w") as f:
        f.write("filename,class\ntest.png,deep\n")
    
    # Create dataset
    dataset = DentalDataset(root_dir, "train")
    
    assert len(dataset) == 1, f"Expected dataset length 1, got {len(dataset)}"
    
    # Test item retrieval
    image, label = dataset[0]
    assert isinstance(image, Image.Image), "Dataset should return PIL Image"
    assert label == dataset.class_to_idx['deep'], \
        f"Expected label {dataset.class_to_idx['deep']}, got {label}"
    
    # Cleanup
    import shutil
    shutil.rmtree(root_dir)

def test_model_training_step(model):
    """Test model training step."""
    # Create dummy batch
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 4, (batch_size,))
    
    # Setup training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive" 