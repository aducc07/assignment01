import cv2
from matplotlib import pyplot as plt

# Load the low-contrast image
image_path = "low con img.jpg"  # Replace with the path to your low-contrast image
low_contrast_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(low_contrast_image)

# Display the original and enhanced images
plt.subplot(1, 2, 1)
plt.imshow(low_contrast_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image (CLAHE)')

plt.show()
