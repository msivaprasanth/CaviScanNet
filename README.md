# CaviScanNet - Dental Caries Detection and Treatment Recommendation System

CaviScanNet is an AI-powered system for detecting dental caries (cavities) in dental images, classifying their severity, and providing treatment recommendations.

## Project Overview

This repository contains the core code for a dental caries detection and recommendation system. The system consists of three main components:

1. **Detection Module**: Identifies the presence and location of dental caries in images
2. **Classification Module**: Categorizes the severity of detected caries
3. **Recommendation System**: Suggests appropriate treatments based on detection and classification results

## Key Files and Their Functions

### Entry Points

* `dental-caries-detection/run.py`: Main application runner that starts the Flask server and initializes the application
* `dental-caries-detection/app/main.py`: Flask application initialization, sets up the application context
* `dental-caries-detection/app/app.py`: Main application setup, configures application components and services

### Web Interface

* `dental-caries-detection/app/routes.py`: API endpoints and URL routing for the web application
* `dental-caries-detection/app/core/routes.py`: Core route handlers that process requests and return responses
* `dental-caries-detection/app/templates/index.html`: Main user interface for the web application

### Model Components

#### Detection Models (locating caries in dental images)

* `dental-caries-detection/app/models/detection/model.py`: Detection model architecture (based on YOLO)
* `dental-caries-detection/app/models/detection/predict.py`: Inference logic for caries detection
* `dental-caries-detection/app/models/detection/train.py`: Training script for the detection model
* `dental-caries-detection/app/models/detection/dataset.py`: Data handling for preparing training and validation data

#### Classification Models (categorizing severity/type)

* `dental-caries-detection/app/models/classification/model.py`: Classification model architecture
* `dental-caries-detection/app/models/classification/train.py`: Training script for classification model

#### Recommendation System

* `recommendation/model.py`: Recommendation model definition
* `recommendation/train.py`: Training script for the recommendation system
* `recommendation/utils.py`: Utilities for recommendation system, including data preprocessing and evaluation metrics

### Core Utilities

* `dental-caries-detection/app/core/models.py`: Data models and schemas for the application
* `dental-caries-detection/app/core/utils.py`: Helper functions for image processing, data handling, etc.
* `dental-caries-detection/app/core/config.py`: Application configuration settings
* `dental-caries-detection/app/core/recommendations.py`: Treatment recommendation logic

## How It Works

1. The user uploads a dental image through the web interface
2. The detection model identifies potential regions with dental caries
3. The classification model categorizes the severity of each detected region
4. The recommendation system provides treatment suggestions based on the detection and classification results
5. Results are displayed to the user via the web interface

## Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: PyTorch, Computer Vision algorithms
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy, OpenCV

## Setup Instructions

This repository contains only the code files without the large model weights and datasets. To run the application:

1. Clone this repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Contact the repository owner for access to model weights and sample data
4. Run the application: `python dental-caries-detection/run.py`

## Note

This repository includes only the essential code files for the project. The complete dataset and trained model weights are not included due to size constraints. 