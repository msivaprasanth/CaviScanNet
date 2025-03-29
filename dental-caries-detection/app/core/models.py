"""Models for dental caries detection and classification."""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from app.models.detection.model import DentalCariesDetector
from app.models.classification.model import DentalCariesClassifier
from app.core.utils import prepare_image, draw_detections, ensure_rgb
from app.core.recommendations import DentalRecommendationSystem

class IntegratedDentalModel:
    """Integrated model for dental caries detection and classification."""
    
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Load Detection Model
        print("Loading detection model...")
        self.detector = DentalCariesDetector()
        detector_checkpoint = torch.load(config.DETECTION_MODEL_PATH, 
                                      map_location=torch.device('cpu'))
        if 'model_state_dict' in detector_checkpoint:
            self.detector.load_state_dict(detector_checkpoint['model_state_dict'])
        else:
            self.detector.load_state_dict(detector_checkpoint)
        self.detector.eval()

        # Load Classification Model
        print("Loading classification model...")
        self.classifier = DentalCariesClassifier()
        classifier_checkpoint = torch.load(config.CLASSIFICATION_MODEL_PATH,
                                        map_location=torch.device('cpu'))
        if 'model_state_dict' in classifier_checkpoint:
            self.classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
        else:
            self.classifier.load_state_dict(classifier_checkpoint)
        self.classifier.eval()
        
        # Initialize recommendation system
        print("Initializing recommendation system...")
        self.recommendation_system = DentalRecommendationSystem()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector.to(self.device)
        self.classifier.to(self.device)
    
    def get_severity_from_score(self, score):
        """Get severity category from confidence score.
        
        Args:
            score (float): Confidence score
            
        Returns:
            str: Severity category
        """
        if score > self.config.DEEP_THRESHOLD:
            return "deep"
        elif score > self.config.MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "superficial"
    
    def is_border_detection(self, box, width, height, border_threshold=0.01):
        """Check if a detection box is too close to image borders.
        
        Args:
            box (list): Box coordinates [x1, y1, x2, y2]
            width (int): Image width
            height (int): Image height
            border_threshold (float): Threshold for border detection (percentage of image dimension)
            
        Returns:
            bool: True if detection is at border
        """
        x1, y1, x2, y2 = box
        
        # Calculate border thresholds
        x_threshold = width * border_threshold
        y_threshold = height * border_threshold
        
        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Check if box touches borders
        is_border = (
            x1 <= x_threshold or  # Left border
            y1 <= y_threshold or  # Top border
            x2 >= width - x_threshold or  # Right border
            y2 >= height - y_threshold  # Bottom border
        )
        
        # Check if box is too large
        is_too_large = (
            box_width > width * 0.15 or  # Box width > 15% of image width
            box_height > height * 0.15    # Box height > 15% of image height
        )
        
        return is_border or is_too_large

    def process_image(self, image):
        """Process image through both models.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            tuple: (result_image, classification, probabilities, detections, recommendations)
        """
        # Ensure image is RGB
        image = ensure_rgb(image)
        
        # Classification prediction
        classifier_input = prepare_image(image, (224, 224))
        classifier_input = classifier_input.to(self.device)
        
        with torch.no_grad():
            classifier_output = self.classifier(classifier_input)
            probabilities = torch.softmax(classifier_output, dim=1)[0]
            
            # Combine superficial and normal probabilities
            combined_probs = torch.zeros(3)
            combined_probs[0] = probabilities[0]  # deep
            combined_probs[1] = probabilities[1]  # medium
            combined_probs[2] = probabilities[2] + probabilities[3]  # superficial + normal
            
            class_names = ['deep', 'medium', 'superficial']
            classification_result = class_names[combined_probs.argmax().item()]
            class_probs = {name: float(prob) for name, prob in zip(class_names, combined_probs)}
        
        # Only run detection if classification shows caries
        if classification_result != 'superficial':
            # Detection prediction
            detector_input = prepare_image(image, self.config.IMAGE_SIZE)
            detector_input = detector_input.to(self.device)
            
            with torch.no_grad():
                detections = self.detector([detector_input.squeeze(0)])  # Remove batch dimension
            
            # Process detections
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            height, width = image_np.shape[:2]
            result_image = image_np.copy()
            
            boxes = detections[0]['boxes'].cpu().numpy()
            scores = detections[0]['scores'].cpu().numpy()
            
            # Filter detections
            high_conf_indices = scores > self.config.CONFIDENCE_THRESHOLD
            filtered_boxes = boxes[high_conf_indices]
            filtered_scores = scores[high_conf_indices]
            
            # Process detections
            detections_list = []
            
            for box, score in zip(filtered_boxes, filtered_scores):
                x1, y1, x2, y2 = [int(coord * width) for coord in box]
                
                # Skip if box is completely outside image bounds
                if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
                    continue
                
                # Skip if detection is at border
                if self.is_border_detection([x1, y1, x2, y2], width, height):
                    continue
                
                # Clip coordinates to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Add detection
                detections_list.append({
                    'coords': [x1, y1, x2, y2],
                    'score': float(score)
                })
                
                # Draw detection in crimson red with thinner border
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 220), 1)  # Crimson red color, thickness 1
                label = f"Cavity ({score:.2f})"
                # Add text without background
                cv2.putText(result_image, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)  # Crimson red text
            
            # Organize detections
            detections_by_severity = {
                classification_result: detections_list
            }
        else:
            # If no caries detected by classification, return original image
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result_image = image_np.copy()
            detections_by_severity = {}
        
        # Get detection counts
        detection_counts = {
            severity: len(detections)
            for severity, detections in detections_by_severity.items()
        }
        
        # Generate recommendations
        recommendations = self.recommendation_system.get_recommendations(
            severity=classification_result,
            detection_counts=detection_counts,
            confidence=max(class_probs.values())
        )
        
        # Format recommendations
        formatted_recommendations = self.recommendation_system.format_recommendations(recommendations)
        
        return (
            result_image,
            classification_result,
            class_probs,
            detections_by_severity,
            {
                'text': formatted_recommendations,
                'recommendations': recommendations
            }
        ) 