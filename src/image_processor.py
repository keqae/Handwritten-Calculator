import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, target_size=(28, 28), training=False):
        self.target_size = target_size
        self.training = training

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        # check if image is already grayscale (either 2D or 3D with 1 channel)
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # if it's 3D with 1 channel, convert to 2D
            if len(image.shape) == 3:
                return image.squeeze()
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image, 5, 75, 75)

    def binarise(self, image: np.ndarray) -> np.ndarray:
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def invert(self, image: np.ndarray) -> np.ndarray:
        return cv2.bitwise_not(image)

    def normalise(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    def resize(self, image: np.ndarray) -> np.ndarray:
        coords = cv2.findNonZero(image) #find bounding boxes
        x, y, w, h = cv2.boundingRect(coords)
        cropped = image[y:y + h, x:x + w]

        # get dimensions
        h, w = cropped.shape
        target_w, target_h = self.target_size

        # scale and resize
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        return cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def center(self, image: np.ndarray, padding=3) -> np.ndarray:
        target_w, target_h = self.target_size
        h, w = image.shape  # get image shape

        # Calculate maximum possible scaling while maintaining aspect ratio
        scale = min((target_w - 2 * padding) / w, (target_h - 2 * padding) / h)
        if scale < 1:  # Only resize down if needed
            w, h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

        # Create canvas and calculate centered position
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)

        # Calculate centered coordinates
        x_offset = (target_w - w) // 2
        y_offset = (target_h - h) // 2

        # Ensure the image fits within the canvas
        y_offset = max(padding, min(y_offset, target_h - h - padding))
        x_offset = max(padding, min(x_offset, target_w - w - padding))

        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = image
        return canvas

    def segment(self, image: np.ndarray):

        # closing morphological operations
        kernel = np.ones((3, 3), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # vertically project the image to form "peaks and valleys"
        projection = np.sum(image, axis=0)

        # find segmentation boundaries
        segments = []
        in_char = False
        start = 0

        for idx, column_value in enumerate(projection):
            # keep track of where the character started
            if column_value > 0 and not in_char:
                in_char = True
                start = idx

            # end character region if valley detected
            elif column_value == 0 and in_char:
                in_char = False
                end = idx
                char_img = image[:, start:end]
                segments.append((char_img))

        # handle case where last character reaches image boundary
        if in_char:
            char_img = image[:, start:]
            segments.append((char_img))

        return segments


    

    # processing pipeline
    def process(self, image: np.ndarray, debug=False) -> list[np.ndarray] | np.ndarray | None:
        if image is None:
            return None


        processed = []

        try:
            if debug:
                print(
                    f"DEBUG: Initial image shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}"
                )

                # print debugging information for the image processing at every step
                image = self.grayscale(image)
                print(f"DEBUG: After grayscale: shape: {image.shape}, min: {image.min()}, max: {image.max()}")

                image = self.denoise(image)
                print(f"DEBUG: After denoise: shape: {image.shape}, min: {image.min()}, max: {image.max()}")

                image = self.binarise(image)
                print(f"DEBUG: After binarise: shape: {image.shape}, min: {image.min()}, max: {image.max()}")

                image = self.invert(image)
                print(f"DEBUG: After invert: shape: {image.shape}, min: {image.min()}, max: {image.max()}")

                # check if image has any non-zero pixels
                non_zero_count = np.count_nonzero(image)
                print(f"DEBUG: Non-zero pixels after invert: {non_zero_count}")

                if not self.training:
                    for contour in self.segment(image):
                        contour = self.resize(contour)
                        contour = self.center(contour)
                        contour = self.normalise(contour)
                        processed.append(contour)

                elif self.training:
                    image = self.resize(image)
                    print(f"DEBUG: After resize: shape: {image.shape}, min: {image.min()}, max: {image.max()}")

                    image = self.center(image)
                    print(f"DEBUG: After center: shape: {image.shape}, min: {image.min()}, max: {image.max()}")

                    image = self.normalise(image)
                    print(f"DEBUG: After normalise: shape: {image.shape}, min: {image.min()}, max: {image.max()}")

                    image = self.binarise(image)
                    print(f"DEBUG: After binarise: shape: {image.shape}, min: {image.min()}, max: {image.max()}")

                    return image

                return processed

            else:

                # presegmentation algorithms
                image = self.grayscale(image)
                image = self.denoise(image)
                image = self.invert(image)
                image = self.binarise(image)

                if not self.training:
                    for contour in self.segment(image):
                        contour = self.resize(contour)
                        contour = self.center(contour)
                        contour = self.normalise(contour)
                        processed.append(contour)

                elif self.training:
                    """
                    image = self.resize(image)
                    image = self.center(image)
                    """
                    
                    image = self.normalise(image)
                    image = self.binarise(image)

                    return image

                return processed

        except Exception as e:
            print(f"Error processing image: {e}")
            return None
