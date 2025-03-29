"""Test cases for integrated dental caries detection and classification model."""

import os
import pytest
import torch
import numpy as np
from PIL import Image
import cv2

from app.core.models import IntegratedDentalModel
from app.core.config import Config

@pytest.fixture
def model():
    """Create integrated model instance for testing."""
    Config.setup()  # Ensure directories exist
    return IntegratedDentalModel(Config)

@pytest.fixture
def sample_image():
    """Create sample image for testing."""
    # Create a dummy X-ray like image
    image = Image.new('RGB', (800, 800), color='gray')
    return image

def test_model_initialization(model):
    """Test if model initializes correctly."""
    assert hasattr(model, 'detector'), "Model should have detector"
    assert hasattr(model, 'classifier'), "Model should have classifier"
    assert model.detector.training == False, "Detector should be in eval mode"
    assert model.classifier.training == False, "Classifier should be in eval mode"

def test_severity_from_score(model):
    """Test severity classification from confidence scores."""
    assert model.get_severity_from_score(0.8) == "deep", \
        "Score 0.8 should be classified as deep"
    assert model.get_severity_from_score(0.6) == "medium", \
        "Score 0.6 should be classified as medium"
    assert model.get_severity_from_score(0.4) == "superficial", \
        "Score 0.4 should be classified as superficial"

def test_process_image(model, sample_image):
    """Test complete image processing pipeline."""
    # Process image
    result_image, classification, probabilities, detections = model.process_image(sample_image)
    
    # Check result types
    assert isinstance(result_image, np.ndarray), \
        "Result image should be numpy array"
    assert isinstance(classification, str), \
        "Classification should be string"
    assert isinstance(probabilities, dict), \
        "Probabilities should be dictionary"
    assert isinstance(detections, dict), \
        "Detections should be dictionary"
    
    # Check probabilities
    assert set(probabilities.keys()) == {'deep', 'medium', 'superficial'}, \
        "Probabilities should have correct classes"
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6, \
        "Probabilities should sum to 1"
    
    # Check detections format
    assert set(detections.keys()) == {'deep', 'medium', 'superficial'}, \
        "Detections should have correct severity levels"
    
    # Check result image dimensions
    assert result_image.shape[:2] == sample_image.size[::-1], \
        "Result image should maintain input dimensions"

def test_edge_case_handling(model):
    """Test handling of edge cases."""
    # Test with very small image
    small_image = Image.new('RGB', (50, 50))
    result = model.process_image(small_image)
    assert all(isinstance(x, type(result[i])) for i, x in enumerate(result)), \
        "Model should handle small images"
    
    # Test with grayscale image
    gray_image = Image.new('L', (800, 800))
    result = model.process_image(gray_image)
    assert all(isinstance(x, type(result[i])) for i, x in enumerate(result)), \
        "Model should handle grayscale images"

def test_detection_filtering(model, sample_image):
    """Test detection filtering and organization."""
    result_image, _, _, detections = model.process_image(sample_image)
    
    for severity, boxes in detections.items():
        for detection in boxes:
            # Check box format
            assert 'coords' in detection, "Detection should have coordinates"
            assert 'score' in detection, "Detection should have confidence score"
            assert len(detection['coords']) == 4, "Box should have 4 coordinates"
            
            # Check coordinate validity
            x1, y1, x2, y2 = detection['coords']
            assert x1 <= x2, "x1 should be <= x2"
            assert y1 <= y2, "y1 should be <= y2"
            assert all(0 <= coord <= max(sample_image.size) for coord in [x1, y1, x2, y2]), \
                "Coordinates should be within image bounds"
            
            # Check score validity
            assert 0 <= detection['score'] <= 1, "Score should be between 0 and 1"

def test_model_consistency(model, sample_image):
    """Test consistency of model predictions."""
    # Run prediction twice
    result1 = model.process_image(sample_image)
    result2 = model.process_image(sample_image)
    
    # Check classification consistency
    assert result1[1] == result2[1], \
        "Classification should be consistent for same image"
    
    # Check probability consistency
    assert all(abs(result1[2][k] - result2[2][k]) < 1e-6 
              for k in result1[2].keys()), \
        "Probabilities should be consistent for same image"
    
    # Check detection consistency
    assert len(result1[3]) == len(result2[3]), \
        "Number of detections should be consistent"

def test_error_handling(model):
    """Test error handling for invalid inputs."""
    # Test with invalid image
    with pytest.raises(Exception):
        model.process_image(None)
    
    # Test with empty image
    empty_image = Image.new('RGB', (0, 0))
    with pytest.raises(Exception):
        model.process_image(empty_image)
    
    # Test with corrupted image
    corrupted_image = np.random.rand(100, 100, 5)  # Invalid number of channels
    with pytest.raises(Exception):
        model.process_image(corrupted_image) 