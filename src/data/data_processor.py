from dataclasses import dataclass
import numpy as np


@dataclass
class DataProcessor:

    def run(self, data) -> tuple:
        train_images, train_labels, test_images, test_labels = (
            self._split_data_into_train_and_test_images(data)
        )
        train_images = self._normalize_images(train_images)
        test_images = self._normalize_images(test_images)
        train_images = self._reshape_images(train_images)
        test_images = self._reshape_images(test_images)
        return train_images, train_labels, test_images, test_labels

    def _split_data_into_train_and_test_images(self, data: tuple) -> tuple:
        train_images, train_labels = data[0]
        test_images, test_labels = data[1]
        return train_images, train_labels, test_images, test_labels

    def _normalize_images(self, images: np.ndarray) -> np.ndarray:
        # Pixel values in images are typically in the range of 0 to 255
        # This scales the pixel values to a range between 0 and 1
        # This can help with numerical stability and convergence during training.
        return images / 255.0

    def _reshape_images(self, images: np.ndarray) -> np.ndarray:
        # The shape of the images should be (num_images, height, width, num_channels)
        # -1: This is a placeholder to automatically calculate the number of images
        # 28: The height of the image
        # 28: The width of the image
        # 1: The number of color channels in the image (1 for grayscale images)
        # The images are reshaped to 4D arrays, as required by Keras
        return images.reshape((-1, 28, 28, 1))
