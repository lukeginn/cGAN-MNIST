from keras.models import Model
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
    Dropout,
    SpatialDropout2D,
    LeakyReLU,
    Reshape,
    Input,
    Embedding,
    Concatenate,
)
from tensorflow_addons.layers import SpectralNormalization
from keras.optimizers import Adam
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Discriminator:
    n_classes: int = 10
    model: Model = field(init=False)

    def __post_init__(self):
        self.model = self.build_model()

    def build_model(self) -> Model:
        image_input, label_input = self._create_inputs()
        label_branch = self._create_label_branch(label_input)
        neural_net = self._merge_branches(image_input, label_branch)
        neural_net = self._add_conv_layers(neural_net)
        neural_net = self._add_dense_layers(neural_net)
        model = Model([image_input, label_input], neural_net)
        self._compile_model(model)
        return model

    def get_model(self) -> Model:
        """Return the compiled model."""
        return self.model

    def _create_inputs(self) -> Tuple[Input, Input]:
        """Create input layers for the image and label branches."""
        image_input = Input(shape=(28, 28, 1))
        label_input = Input(shape=(1,))
        return image_input, label_input

    def _create_label_branch(self, label_input: Input) -> Model:
        """Create the label branch of the discriminator."""
        label_branch = Embedding(self.n_classes, 50)(label_input)
        label_branch = Dense(28 * 28)(label_branch)
        label_branch = Reshape((28, 28, 1))(label_branch)
        return label_branch

    def _merge_branches(self, image_input: Input, label_branch: Model) -> Model:
        """Merge the image and label branches."""
        return Concatenate()([image_input, label_branch])

    def _add_conv_layers(self, neural_net: Model) -> Model:
        """Add convolutional layers to the neural network."""
        neural_net = SpectralNormalization(Conv2D(128, (3, 3)))(neural_net)
        neural_net = LeakyReLU(alpha=0.2)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = MaxPooling2D((2, 2))(neural_net)
        neural_net = SpatialDropout2D(0.5)(neural_net)
        neural_net = SpectralNormalization(Conv2D(128, (3, 3)))(neural_net)
        neural_net = SpectralNormalization(Conv2D(128, (3, 3)))(neural_net)
        neural_net = LeakyReLU(alpha=0.2)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = MaxPooling2D((2, 2))(neural_net)
        neural_net = SpatialDropout2D(0.5)(neural_net)
        neural_net = Flatten()(neural_net)
        return neural_net

    def _add_dense_layers(self, neural_net: Model) -> Model:
        """Add dense layers to the neural network."""
        neural_net = SpectralNormalization(Dense(600))(neural_net)
        neural_net = LeakyReLU(alpha=0.2)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = Dropout(0.5)(neural_net)
        neural_net = SpectralNormalization(Dense(400))(neural_net)
        neural_net = LeakyReLU(alpha=0.2)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = Dropout(0.5)(neural_net)
        neural_net = SpectralNormalization(Dense(200))(neural_net)
        neural_net = LeakyReLU(alpha=0.2)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = Dropout(0.5)(neural_net)
        neural_net = SpectralNormalization(Dense(20))(neural_net)
        neural_net = LeakyReLU(alpha=0.2)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = Dropout(0.5)(neural_net)
        neural_net = SpectralNormalization(Dense(1, activation="sigmoid"))(neural_net)
        return neural_net

    def _compile_model(self, model: Model) -> None:
        """Compile the model with the specified optimizer, loss, and metrics."""
        model.compile(
            optimizer=Adam(lr=0.0001, beta_1=0.5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.trainable = True
