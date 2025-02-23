from dataclasses import dataclass, field
from keras.datasets.mnist import load_data


@dataclass
class DataReader:
    data: tuple = field(default=None, init=False)

    def load_data(self):
        # Load the MNIST dataset
        # The dataset is already split into training and testing sets
        # The training set has 60,000 images, and the testing set has 10,000 images
        # Each image is a 28x28 grayscale image
        # The pixel values are between 0 and 255
        # The labels are integers between 0 and 9
        # The data is loaded as NumPy arrays
        self.data = load_data()
        return self.data
