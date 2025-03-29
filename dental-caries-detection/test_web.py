"""Script to test the dental caries detection web application."""

import os
import requests
import time
from PIL import Image
import matplotlib.pyplot as plt
import json

def test_web_app():
    """Test the web application with sample images."""
    # Server URL
    BASE_URL = 'http://localhost:5000'
    
    # Create test directory
    os.makedirs('test_results', exist_ok=True)
    
    def analyze_image(image_path):
        """Analyze a single image."""
        print(f"\nAnalyzing image: {os.path.basename(image_path)}")
        
        # Check if server is running
        try:
            response = requests.get(BASE_URL)
            if response.status_code != 200:
                print("Error: Server is not responding correctly")
                return
        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to server. Make sure it's running.")
            return
        
        # Prepare image for upload
        with open(image_path, 'rb') as f:
            files = {'file': f}
            
            # Send request
            try:
                start_time = time.time()
                response = requests.post(f'{BASE_URL}/api/analyze', files=files)
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Print results
                    print("\nResults:")
                    print(f"Processing time: {processing_time:.2f} seconds")
                    print(f"Classification: {data['classification']['predicted_class']}")
                    print("\nClass Scores:")
                    for cls, score in data['classification']['class_scores'].items():
                        print(f"  {cls}: {score:.2%}")
                    print("\nDetections:")
                    for severity, detections in data['detections'].items():
                        print(f"  {severity}: {len(detections)} detections")
                        for detection in detections:
                            print(f"    Score: {detection['score']:.2%}")
                    
                    # Download and save result image
                    result_image_url = f"{BASE_URL}{data['result_image']}"
                    result_image = requests.get(result_image_url)
                    if result_image.status_code == 200:
                        output_path = os.path.join(
                            'test_results',
                            f'result_{os.path.basename(image_path)}'
                        )
                        with open(output_path, 'wb') as f:
                            f.write(result_image.content)
                        print(f"\nResult image saved to: {output_path}")
                        
                        # Display original and result images
                        plt.figure(figsize=(15, 5))
                        
                        # Original image
                        plt.subplot(1, 2, 1)
                        img = Image.open(image_path)
                        plt.imshow(img)
                        plt.title('Original Image')
                        plt.axis('off')
                        
                        # Result image
                        plt.subplot(1, 2, 2)
                        result_img = Image.open(output_path)
                        plt.imshow(result_img)
                        plt.title('Detection Results')
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.show()
                        plt.close()  # Close the figure to free memory
                    
                else:
                    print(f"Error: {response.json().get('error', 'Unknown error')}")
            
            except Exception as e:
                print(f"Error during analysis: {str(e)}")

    # Get all PNG files from the test images directory
    test_images_dir = 'data/test_images'
    test_images = [
        os.path.join(test_images_dir, f) 
        for f in os.listdir(test_images_dir) 
        if f.endswith('.png')
    ]
    
    # Sort images by name for consistent ordering
    test_images.sort()
    
    print(f"\nFound {len(test_images)} test images")
    
    # Ask user how many images to process
    print("\nHow many images would you like to process?")
    print(f"Available range: 1-{len(test_images)}")
    try:
        num_images = int(input("Enter number (default: 3): ") or "3")
        num_images = min(max(1, num_images), len(test_images))
    except ValueError:
        num_images = 3
        print("Invalid input, using default value: 3")
    
    # Process selected number of images
    for image_path in test_images[:num_images]:
        if os.path.exists(image_path):
            analyze_image(image_path)
        else:
            print(f"\nWarning: Test image not found: {image_path}")

if __name__ == '__main__':
    print("Starting web application test...")
    print("Make sure the server is running (python run.py)")
    print("Testing will begin in 5 seconds...")
    time.sleep(5)
    test_web_app() 