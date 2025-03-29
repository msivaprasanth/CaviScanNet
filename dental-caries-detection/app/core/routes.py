"""Route configuration for the application."""

import os
from datetime import datetime
from flask import render_template, request, url_for
from PIL import Image
import cv2
from app.core.models import IntegratedDentalModel
from app.core.utils import allowed_file
from app.core.config import Config

# Initialize model with config
model = IntegratedDentalModel(Config)

def configure_routes(app):
    """Configure routes for the application."""
    
    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        """Handle file upload and image processing."""
        if request.method == 'POST':
            if 'file' not in request.files:
                return 'No file uploaded'
            
            file = request.files['file']
            if file.filename == '':
                return 'No file selected'
            
            if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                # Read and process the image
                image = Image.open(file.stream)
                result_image, predicted_class, class_scores, detections, recommendations = model.process_image(image)
                
                # Save the result image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_path = f'results/result_{timestamp}.png'
                os.makedirs(os.path.dirname(os.path.join(app.static_folder, result_path)), 
                          exist_ok=True)
                cv2.imwrite(os.path.join(app.static_folder, result_path), result_image)
                
                return render_template('index.html',
                                     result_image=url_for('static', filename=result_path),
                                     predicted_class=predicted_class,
                                     class_scores=class_scores,
                                     detections_by_severity=detections,
                                     recommendations=recommendations)
            
            return 'Invalid file type'
        
        return render_template('index.html')
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze_image():
        """API endpoint for image analysis."""
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            try:
                # Read and process the image
                image = Image.open(file.stream)
                result_image, predicted_class, class_scores, detections, recommendations = model.process_image(image)
                
                # Save the result image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_path = f'results/result_{timestamp}.png'
                os.makedirs(os.path.dirname(os.path.join(app.static_folder, result_path)), 
                          exist_ok=True)
                cv2.imwrite(os.path.join(app.static_folder, result_path), result_image)
                
                # Format response
                response = {
                    'success': True,
                    'classification': {
                        'predicted_class': predicted_class,
                        'class_scores': class_scores
                    },
                    'detections': detections,
                    'recommendations': recommendations,
                    'result_image': url_for('static', filename=result_path)
                }
                
                return response
            
            except Exception as e:
                return {'error': str(e)}, 500
        
        return {'error': 'Invalid file type'}, 400 