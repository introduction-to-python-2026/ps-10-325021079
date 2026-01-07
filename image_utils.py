from PIL import Image
import numpy as np
from scipy.signal import convolve2d

from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball

def load_image(file_path):
    """
    Reads a color image from the given file path and returns it as a NumPy array.
    """
    try:
        # Open the image using Pillow
        with Image.open(file_path) as img:
            # Ensure the image is in RGB mode (handling potential transparency/grayscale issues)
            img = img.convert('RGB')
            # Convert the image object to a NumPy array
            image_array = np.array(img)
            return image_array
            
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def edge_detection(image_array):
    """
    Performs edge detection on a given image array using Sobel operators.
    
    """
    
    # 1. Convert to Grayscale
    # Average the 3 color channels (axis 2) to get a 2D array
    grayscale_image = np.mean(image_array, axis=2)

    # 2. Define the Kernels
    # Filter for vertical changes (Gradient in Y direction)
    kernelY = np.array([
        [ 1,  1,  1],
        [ 0,  0,  0],
        [-1, -1, -1]
    ])
    
    # Filter for horizontal changes (Gradient in X direction)
    kernelX = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # 3. Apply Convolution
    # mode='same' ensures output size matches input size
    # boundary='fill' with fillvalue=0 ensures zero padding
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 4. Compute Magnitude
    # Calculate the hypotenuse of the X and Y gradients
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
