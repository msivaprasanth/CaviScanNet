# Testing the Dental Caries Detection System

This document provides instructions for testing the web application.

## Prerequisites

1. Install all dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the model checkpoints in place:
```
app/models/classification/best_model.pth
app/models/detection/model_epoch_25.pth
```

## Running the Web Application

1. Start the Flask server:
```bash
python run.py
```

The server will start at `http://localhost:5000`

## Testing Methods

### 1. Web Interface Testing

1. Open your browser and navigate to `http://localhost:5000`
2. Use the upload interface to select and analyze dental X-ray images
3. View the results directly in the browser

### 2. Automated Testing

Run the automated test suite:
```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_app.py
pytest tests/test_classification.py
pytest tests/test_detection.py
pytest tests/test_integration.py
```

### 3. Interactive Testing Script

1. Place your test images in the `data/test_images/` directory
2. Run the testing script:
```bash
python test_web.py
```

This script will:
- Test server connectivity
- Process each test image
- Display results and timing information
- Save and display result images
- Show detection visualizations

## Test Cases

### Basic Functionality
- Upload and process dental X-ray images
- View detection results and classifications
- Download result images

### Edge Cases
- Large images (>4000x4000 pixels)
- Small images (<100x100 pixels)
- Grayscale images
- Invalid file types
- Empty files
- Corrupted images

### Performance Testing
- Multiple concurrent requests
- Processing time measurement
- Memory usage monitoring

## Expected Results

For each analyzed image, you should see:

1. Classification Result:
   - Overall severity classification
   - Confidence scores for each class

2. Detection Results:
   - Bounding boxes around detected caries
   - Severity labels for each detection
   - Confidence scores

3. Visualization:
   - Original image
   - Annotated image with detections
   - Color-coded severity indicators

## Troubleshooting

### Common Issues

1. Server Connection Error
```
Error: Cannot connect to server
Solution: Make sure the Flask server is running (python run.py)
```

2. Model Loading Error
```
Error: Model checkpoint not found
Solution: Verify model files are in correct locations
```

3. Memory Issues
```
Error: Out of memory
Solution: Reduce batch size or image dimensions
```

### Performance Optimization

If experiencing slow processing:
1. Reduce image size before upload
2. Use batch processing for multiple images
3. Enable GPU acceleration if available

## Reporting Issues

When reporting issues, please include:
1. Test image used
2. Error message or unexpected behavior
3. System specifications
4. Steps to reproduce the issue

## Continuous Integration

The project includes GitHub Actions workflows for:
1. Running tests on each push
2. Code quality checks
3. Docker image builds

## Security Testing

1. Input Validation
   - File type verification
   - Size limits
   - Content validation

2. Error Handling
   - Graceful error messages
   - Proper HTTP status codes
   - Secure error details

3. Resource Management
   - Temporary file cleanup
   - Memory management
   - Connection handling 