import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_defect_density(binary_image):
    # Calculate the area of foreground (defect) and background (non-defect) regions in the binary image
    total_area = binary_image.size
    defect_area = np.count_nonzero(binary_image)  # Count of non-zero pixels (white pixels)

    # Determine which area is smaller (defective area)
    if defect_area > total_area / 2:
        defect_area = total_area - defect_area  # Use the smaller area as the defective area

    # Calculate defect density
    if total_area > 0:
        defect_density = defect_area / total_area
    else:
        defect_density = 0.0
    
    return defect_density

def defect_detect(image):
    img = image.copy()
    # Preprocess the input image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    blr = cv2.blur(v, (15, 15))
    dst = cv2.fastNlMeansDenoising(blr, None, 10, 7, 21)
    _, binary = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate defect density based on the binary image
    defect_density = calculate_defect_density(binary)
    
    return binary, defect_density

# Path to the input images folder
input_folder = 'INPUT_images'

# Process each image in the folder
for filename in os.listdir(input_folder):
    # Read the image
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    
    if image is None:
        continue
    
    # Detect defects in the image
    binary_image, defect_density = defect_detect(image)
    
    # Display the original image and processed binary image
    plt.figure(figsize=(12, 6))
    
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Fabric Image')
    plt.axis('off')
    
    # Display the processed binary image
    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binary Image\nDefect Density: {defect_density:.4f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
