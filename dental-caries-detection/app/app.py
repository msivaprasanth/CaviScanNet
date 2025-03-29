"""Flask application for dental caries detection."""

import os
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np

from app.core.config import Config
from app.core.models import IntegratedDentalModel
from app.core.utils import ensure_rgb

app = Flask(__name__)
app.config.from_object(Config)

# Initialize model
model = None

def init_model():
    """Initialize the integrated model."""
    global model
    if model is None:
        Config.setup()  # Ensure directories exist
        model = IntegratedDentalModel(Config)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze dental X-ray image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'Invalid file'}), 400
    
    # Check file extension
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Initialize model if needed
        init_model()
        
        # Read and process image
        image = Image.open(file.stream)
        image = ensure_rgb(image)
        
        # Process image through model
        result_image, classification, probabilities, detections = model.process_image(image)
        
        # Save result image
        result_filename = f'result_{os.path.splitext(file.filename)[0]}.png'
        result_path = os.path.join(Config.RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)
        
        # Count detections by severity
        detection_counts = {
            severity: len(boxes) for severity, boxes in detections.items()
        }
        
        return jsonify({
            'success': True,
            'classification': classification,
            'probabilities': probabilities,
            'detection_counts': detection_counts,
            'result_image': f'/results/{result_filename}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def results(filename):
    """Serve result images."""
    return send_from_directory(Config.RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True) 