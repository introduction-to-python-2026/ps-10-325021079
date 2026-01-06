import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

# Import your functions from the file where you saved them
# Ensure 'image_utils.py' is in the same folder as this script
from image_utils import load_image, edge_detection

# 1. Load the image
image_path = 'LamaShenhav.JPG'
image = load_image(image_path)

if image is not None:
    print(f"Image loaded successfully. Shape: {image.shape}")

    # 2. Suppress noise using a median filter
    # We use ball(1) to fit the 3 RGB channels (size 3x3x3).
    # ball(3) would be too large for the color channels.
    print("Applying median filter to remove noise...")
    clean_image = median(image, ball(1))

    # 3. Apply Edge Detection
    print("Detecting edges...")
    edgeMAG = edge_detection(clean_image)

    # 4. Analyze Histogram to find a Threshold
    # We flatten the array to 1D to plot the histogram of all pixel values
    plt.figure(figsize=(10, 4))
    plt.hist(edgeMAG.flatten(), bins=50, color='gray', log=True)
    plt.title("Histogram of Edge Magnitudes")
    plt.xlabel("Pixel Intensity (Edge Strength)")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 5. Thresholding
    # Based on the histogram, pick a value that separates the 'noise' (low values)
    # from the 'edges' (high values).
    # You can adjust this value after looking at the histogram above.
    threshold = 80

    # Create binary array: True (1) where edges are strong, False (0) otherwise
    edge_binary = edgeMAG > threshold

    # 6. Display and Save the Result
    plt.figure(figsize=(10, 5))
    plt.imshow(edge_binary, cmap='gray')
    plt.title(f"Binary Edge Detection (Threshold={threshold})")
    plt.axis('off')
    plt.show()

    # Convert boolean array (True/False) to Integer (0/255) for saving
    # uint8 is the standard format for images
    final_image_data = (edge_binary * 255).astype(np.uint8)

    # Save using PIL
    edge_image = Image.fromarray(final_image_data)
    save_path = 'my_edges.png'
    edge_image.save(save_path)
    print(f"Success! Edge image saved to {save_path}")

else:
    print("Failed to load image. Please check the file path.")
