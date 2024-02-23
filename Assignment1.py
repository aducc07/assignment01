import cv2
import numpy as np
from matplotlib import pyplot as plt

def analyze_image(image_path, target_size=(300, 300), threshold_value=128):
    # Load image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize image
    resized_image = cv2.resize(original_image, target_size)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(resized_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Display the original and processed images
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(resized_image, cmap='gray')
    plt.title(f'Resized to {target_size}')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binary Thresholding (Threshold={threshold_value})')

    plt.show()

# Example Usage
image_path = "mu.jpg"  # Replace with the path to your image
analyze_image(image_path)
