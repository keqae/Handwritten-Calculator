import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import cv2
from pathlib import Path
import matplotlib
import os


matplotlib.use("Agg")

# calculate the project root
project_root = Path(__file__).resolve().parent.parent.parent  # adjust parent count based on actual depth
sys.path.insert(0, str(project_root))

# now import
from src.image_processor import ImageProcessor


#callable function to visualise without returns
def visualise(img: np.ndarray | tf.Tensor, function: str) -> None:
    # check if its a numpy array, if not, convert it to one
    if isinstance(img, tf.Tensor):
        img = img.numpy()


    resized = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA) # resize image

    if resized.dtype != np.uint8:
        # normalize if needed (if values are in [0,1] range)
        if resized.max() <= 1.0:
            resized = (resized * 255).astype(np.uint8)
        else:
            resized = resized.astype(np.uint8)

    #plt.imshow(resized) # show image before function

    try:
        if function == "segment": # handle segment

            #make folder to save all segmented contours in
            os.makedirs("data/dataset/expressions_segmented", exist_ok=True)

            processor = ImageProcessor(training=True)

            segments = processor.segment(img)

            for idx, image in enumerate(segments):
                resized = cv2.resize(image[0], (360, 360), interpolation=cv2.INTER_AREA)

                plt.imshow(image[0], cmap="gray")
                plt.axis("off")

                plt.savefig(f"data/dataset/expressions_segmented/{idx}.png")
            

        else:
            plt.imshow(cv2.resize(getattr(ImageProcessor(training=True), function)(resized), (360, 360), interpolation=cv2.INTER_AREA)) # show image after

            plt.axis("off")
            plt.savefig(f"data/visualiser_output/{function}.png")
        
        return

    except AttributeError: # if cant find the function, raise error
        raise AttributeError(f"Function '{function}' not found.")


def visualise_projection(image: np.ndarray) -> None:
    # vertically project the image
    processor = ImageProcessor(training=True)
    
    projection = processor.segment(image)

    # plot the projection
    plt.bar(range(len(projection)), projection)

    plt.axis("off")
    plt.savefig("data/dataset/expressions_segmented/image.png")
