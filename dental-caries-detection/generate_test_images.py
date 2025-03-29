"""Script to generate sample dental X-ray images for testing."""

import os
import numpy as np
from PIL import Image, ImageDraw

def create_sample_xray(size=(800, 800), has_caries=True):
    """Create a sample dental X-ray image.
    
    Args:
        size (tuple): Image size (width, height)
        has_caries (bool): Whether to add simulated caries
        
    Returns:
        PIL.Image: Generated image
    """
    # Create base image (gray background)
    base = np.random.randint(180, 220, size=(*size, 3), dtype=np.uint8)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    
    # Add tooth-like structures
    for i in range(3):
        # Draw tooth outline
        x1 = size[0] // 4 + i * (size[0] // 3)
        y1 = size[1] // 4
        x2 = x1 + size[0] // 4
        y2 = y1 + size[1] // 2
        
        # Make tooth shape slightly irregular
        points = [
            (x1, y1 + (y2-y1)//4),
            (x1 + (x2-x1)//4, y1),
            (x2 - (x2-x1)//4, y1),
            (x2, y1 + (y2-y1)//4),
            (x2, y2 - (y2-y1)//4),
            (x2 - (x2-x1)//4, y2),
            (x1 + (x2-x1)//4, y2),
            (x1, y2 - (y2-y1)//4)
        ]
        draw.polygon(points, fill=(200, 200, 200), outline=(150, 150, 150))
        
        if has_caries:
            # Add simulated caries (darker regions)
            caries_x = x1 + np.random.randint(10, x2-x1-10)
            caries_y = y1 + np.random.randint(10, y2-y1-10)
            caries_size = np.random.randint(20, 40)
            draw.ellipse(
                [caries_x, caries_y,
                 caries_x + caries_size, caries_y + caries_size],
                fill=(100, 100, 100)
            )
    
    return image

def generate_test_images():
    """Generate test images and save them."""
    # Create output directory
    os.makedirs('data/test_images', exist_ok=True)
    
    # Generate different types of images
    print("Generating test images...")
    
    # 1. Normal size image with caries
    image1 = create_sample_xray(size=(800, 800), has_caries=True)
    image1.save('data/test_images/sample1.png')
    print("Generated sample1.png - Normal size with caries")
    
    # 2. Large image with caries
    image2 = create_sample_xray(size=(2000, 2000), has_caries=True)
    image2.save('data/test_images/sample2.png')
    print("Generated sample2.png - Large size with caries")
    
    # 3. Normal size image without caries
    image3 = create_sample_xray(size=(800, 800), has_caries=False)
    image3.save('data/test_images/sample3.png')
    print("Generated sample3.png - Normal size without caries")
    
    print("\nTest images have been generated in data/test_images/")

if __name__ == '__main__':
    generate_test_images() 