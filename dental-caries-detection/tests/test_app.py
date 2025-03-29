"""Test cases for Flask application."""

import os
import pytest
import json
from PIL import Image
import io
import numpy as np
from werkzeug.datastructures import FileStorage

from app.app import app
from app.core.config import Config

@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image():
    """Create sample image for testing."""
    image = Image.new('RGB', (800, 800), color='gray')
    byte_io = io.BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return FileStorage(
        stream=byte_io,
        filename='test.png',
        content_type='image/png'
    )

def test_index_route(client):
    """Test index route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Dental Caries Analysis System' in response.data

def test_analyze_no_file(client):
    """Test analyze route without file."""
    response = client.post('/analyze')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'No file uploaded'

def test_analyze_empty_file(client):
    """Test analyze route with empty file."""
    response = client.post('/analyze', data={
        'file': (io.BytesIO(), '')
    })
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'Invalid file'

def test_analyze_invalid_extension(client):
    """Test analyze route with invalid file extension."""
    image = Image.new('RGB', (800, 800))
    byte_io = io.BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    response = client.post('/analyze', data={
        'file': (byte_io, 'test.txt')
    })
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == 'Invalid file type'

def test_analyze_valid_image(client, sample_image):
    """Test analyze route with valid image."""
    response = client.post('/analyze', data={
        'file': sample_image
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check response format
    assert 'success' in data
    assert data['success'] == True
    assert 'classification' in data
    assert 'probabilities' in data
    assert 'detection_counts' in data
    assert 'result_image' in data
    
    # Check probabilities
    probs = data['probabilities']
    assert set(probs.keys()) == {'deep', 'medium', 'superficial'}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    
    # Check detection counts
    counts = data['detection_counts']
    assert set(counts.keys()) == {'deep', 'medium', 'superficial'}
    assert all(isinstance(v, int) for v in counts.values())

def test_results_route(client, sample_image):
    """Test results route."""
    # First analyze an image to create a result
    response = client.post('/analyze', data={
        'file': sample_image
    })
    data = json.loads(response.data)
    result_path = data['result_image'].split('/')[-1]
    
    # Then try to retrieve the result
    response = client.get(f'/results/{result_path}')
    assert response.status_code == 200
    assert response.mimetype == 'image/png'

def test_results_invalid_file(client):
    """Test results route with invalid filename."""
    response = client.get('/results/nonexistent.png')
    assert response.status_code == 404

def test_concurrent_requests(client, sample_image):
    """Test handling of concurrent requests."""
    import threading
    
    def make_request():
        response = client.post('/analyze', data={
            'file': sample_image
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
    
    # Create multiple threads
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

def test_large_image(client):
    """Test handling of large images."""
    # Create a large image
    large_image = Image.new('RGB', (4000, 4000))
    byte_io = io.BytesIO()
    large_image.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    response = client.post('/analyze', data={
        'file': (byte_io, 'large.png')
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True

def test_model_initialization(client, sample_image):
    """Test model initialization on first request."""
    import app.app
    
    # Store original model
    original_model = app.app.model
    
    try:
        # Clear model
        app.app.model = None
        
        # Make request
        response = client.post('/analyze', data={
            'file': sample_image
        })
        assert response.status_code == 200
        
        # Check if model was initialized
        assert app.app.model is not None
        
    finally:
        # Restore original model
        app.app.model = original_model

def test_error_handling(client):
    """Test error handling in various scenarios."""
    # Test with corrupted image
    corrupted_data = b'not an image'
    response = client.post('/analyze', data={
        'file': (io.BytesIO(corrupted_data), 'test.png')
    })
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data

    # Test with zero-size image
    empty_image = Image.new('RGB', (0, 0))
    byte_io = io.BytesIO()
    empty_image.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    response = client.post('/analyze', data={
        'file': (byte_io, 'empty.png')
    })
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data 