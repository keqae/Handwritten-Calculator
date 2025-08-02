import os
import cv2
import numpy as np
import sys
from pathlib import Path
import random
import csv

# Calculate the project root (parent of the src directory)
project_root = Path(__file__).resolve().parent.parent.parent  # Go up to src then to project root

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Add src directory to path

# Import from correct location
from image_processor import ImageProcessor


def load(image_path: str, save_path: str = "dataset/segmented.csv", testing=False):
    """
    Load an image, process it through the image processor, and save the segmented
    results to a CSV file.

    Args:
        image_path: Path to the image to process
        save_path: Path where to save the CSV file
        testing: If True, only one random sample will be saved
    """
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return False

    # Load image as nparray
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image at {image_path}")
        return False

    # Create full path using project root
    full_save_path = project_root / save_path

    # Ensure the directory exists
    full_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize processor and process image
    processor = ImageProcessor()
    processed = processor.process(img)

    if not processed or len(processed) == 0:
        print("No symbols were detected in the image")
        return False

    # Save the processed images to CSV
    with open(full_save_path, "w", newline='') as file:
        csv_writer = csv.writer(file)
        # Write header
        csv_writer.writerow(["symbol_id", "image_data", "image_shape"])

        if testing:
            # Randomly select an element from processed
            index = random.randint(0, len(processed) - 1)
            symbol = processed[index]
            # Convert the image to a flattened string representation
            flattened = symbol.flatten().tolist()
            csv_writer.writerow([index, str(flattened), str(symbol.shape)])
            print(f"Symbol {index} saved for testing")
        else:
            for i, symbol in enumerate(processed):
                # Convert the image to a flattened string representation
                flattened = symbol.flatten().tolist()
                csv_writer.writerow([i, str(flattened), str(symbol.shape)])
            print(f"All {len(processed)} symbols saved to {full_save_path}")

    return True