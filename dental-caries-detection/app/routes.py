"""API routes for dental caries detection application."""

import os
import time
import cv2
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from app.core.models import IntegratedDentalModel
from app.core.utils import allowed_file
from app.config import Config

# Initialize blueprint
api = Blueprint('api', __name__)

# Initialize model
model = IntegratedDentalModel(Config)

@api.route('/analyze', methods=['POST'])
def analyze():
    """Analyze dental X-ray image for caries detection.
    
    Returns:
        JSON response with analysis results
    """
    # Start timing
    start_time = time.time()
    
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Read and process image
        image = Image.open(file.stream)
        result_image, classification, probabilities, detections, recommendations = model.process_image(image)
        
        # Save result image
        filename = secure_filename(file.filename)
        base_name = os.path.splitext(filename)[0]
        result_path = os.path.join(Config.RESULT_FOLDER, f'result_{base_name}.png')
        cv2.imwrite(result_path, result_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            'success': True,
            'processing_time': processing_time,
            'classification': {
                'result': classification,
                'probabilities': probabilities
            },
            'detections': detections,
            'recommendations': recommendations,
            'result_image': f'/results/result_{base_name}.png'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/results/<path:filename>')
def results(filename):
    """Serve result images."""
    return send_file(os.path.join(Config.RESULT_FOLDER, filename))

@api.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}) 