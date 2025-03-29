"""Recommendation system integration for dental caries detection."""

import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from typing import List, Dict, Any

class DentalRecommendationSystem:
    """Recommendation system for dental care advice."""
    
    def __init__(self):
        """Initialize the recommendation system."""
        self.templates = self.create_recommendation_templates()
        
    def create_recommendation_templates(self) -> Dict[int, Dict[str, Any]]:
        """Create recommendation templates."""
        return {
            0: {
                'text': "Schedule regular dental check-ups every 6 months for early detection and prevention. Professional cleaning and examination are essential for maintaining oral health.",
                'severity': 'low',
                'urgency': 'routine',
                'actions': [
                    'Schedule bi-annual check-ups',
                    'Get professional cleaning',
                    'Monitor any changes'
                ]
            },
            1: {
                'text': "Use high-fluoride toothpaste and consider professional fluoride treatment. This helps strengthen tooth enamel and prevent cavity progression.",
                'severity': 'low',
                'urgency': 'preventive',
                'actions': [
                    'Switch to fluoride toothpaste',
                    'Consider fluoride treatment',
                    'Use fluoride mouthwash'
                ]
            },
            2: {
                'text': "Immediate dental visit required for treatment of deep cavity. Delay may lead to more serious complications like infection or abscess.",
                'severity': 'high',
                'urgency': 'immediate',
                'actions': [
                    'Schedule emergency appointment',
                    'Manage pain appropriately',
                    'Avoid pressure on affected area'
                ]
            },
            3: {
                'text': "Modify diet to reduce sugar and acid intake. Limit snacking frequency and choose tooth-friendly foods to prevent cavity progression.",
                'severity': 'medium',
                'urgency': 'important',
                'actions': [
                    'Reduce sugar consumption',
                    'Avoid acidic drinks',
                    'Choose healthy snacks'
                ]
            },
            4: {
                'text': "Improve brushing technique focusing on problem areas. Use proper brushing method, ensure thorough cleaning, and consider using interdental cleaning tools.",
                'severity': 'low',
                'urgency': 'educational',
                'actions': [
                    'Use proper brushing technique',
                    'Implement interdental cleaning',
                    'Brush for full 2 minutes'
                ]
            }
        }
    
    def get_recommendations(self, severity: str, detection_counts: Dict[str, int], confidence: float) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations based on severity and detections.
        
        Args:
            severity (str): Overall severity classification
            detection_counts (dict): Number of detections per severity level
            confidence (float): Model's confidence in classification
            
        Returns:
            list: List of relevant recommendations with actions
        """
        # Map severity to recommendation indices
        severity_recommendations = {
            'normal': [0, 4],  # Regular check-ups and oral hygiene
            'superficial': [1, 4],  # Fluoride treatment and oral hygiene
            'medium': [1, 3, 4],  # Fluoride, diet, and oral hygiene
            'deep': [2, 3]  # Immediate treatment and diet modification
        }
        
        # Get base recommendations for severity
        rec_indices = severity_recommendations.get(severity, [0])
        
        # Add additional recommendations based on detection counts
        if detection_counts.get('deep', 0) > 0:
            rec_indices = list(set(rec_indices + [2]))  # Add immediate treatment
        if detection_counts.get('medium', 0) > 1:
            rec_indices = list(set(rec_indices + [3]))  # Add diet modification
        
        # Get full recommendations with actions
        recommendations = []
        for idx in rec_indices:
            template = self.templates[idx]
            recommendations.append({
                'text': template['text'],
                'urgency': template['urgency'],
                'actions': template['actions'],
                'severity': template['severity']
            })
        
        return recommendations
    
    def format_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """
        Format recommendations into a readable string.
        
        Args:
            recommendations (list): List of recommendation dictionaries
            
        Returns:
            str: Formatted recommendations
        """
        formatted = []
        for rec in recommendations:
            # Add main recommendation
            formatted.append(f"• {rec['text']}")
            
            # Add specific actions
            for action in rec['actions']:
                formatted.append(f"  - {action}")
            
            formatted.append("")  # Add blank line between recommendations
        
        return "\n".join(formatted) 