
import numpy as np
import sys
import tensorflow as tf
import cv2
from pathlib import Path


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

    cv2.imshow("Visualising Before:", resized) # show image before function

    try:
        if function == "segment": # handle segment
            for image in getattr(ImageProcessor(training=True), function)(resized):
                cv2.imshow("Visualising After:", cv2.resize(image, (360, 360), interpolation=cv2.INTER_AREA))

        else:
            cv2.imshow("Visualising After:", cv2.resize(getattr(ImageProcessor(training=True), function)(resized), (360, 360), interpolation=cv2.INTER_AREA)) # show image after

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    except AttributeError: # if cant find the function, raise error
        raise AttributeError(f"Function '{function}' not found.")
