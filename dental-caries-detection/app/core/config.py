"""Configuration settings for the dental caries detection application."""

import os
from pathlib import Path

class Config:
    """Application configuration."""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
    
    # Model paths
    DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'detection', 'model_epoch_25.pth')
    CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'classification', 'best_model.pth')
    
    # Static folder for Flask
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    
    # Image settings
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    IMAGE_SIZE = (800, 800)  # (height, width)
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD = 0.3
    DEEP_THRESHOLD = 0.7
    MEDIUM_THRESHOLD = 0.5
    
    # Edge detection
    EDGE_THRESHOLD = 0.03  # 3% of image dimensions
    
    # Create required directories
    @classmethod
    def setup(cls):
        """Create required directories if they don't exist."""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.RESULTS_FOLDER, exist_ok=True)
        os.makedirs(cls.STATIC_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(cls.STATIC_FOLDER, 'results'), exist_ok=True)
        os.makedirs(os.path.join(cls.STATIC_FOLDER, 'uploads'), exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    # Use temporary directories for testing
    UPLOAD_FOLDER = Path('/tmp/test_uploads')
    RESULTS_FOLDER = Path('/tmp/test_results') 