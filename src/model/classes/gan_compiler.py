from keras.models import Model
from keras.optimizers import Adam
from dataclasses import dataclass


@dataclass
class GANCompiler:
    generator_model: Model
    discriminator_model: Model
    adam_lr: float = 0.0002
    adam_beta_1: float = 0.5
    model: Model = None

    def __post_init__(self):
        self.model = self.build_model()

    def build_model(self) -> Model:
        """Build and compile the GAN model."""
        self.discriminator_model.trainable = False
        gen_latent_input, gen_label_input = self.generator_model.input
        gen_image_output = self.generator_model.output
        dis_prediction = self.discriminator_model([gen_image_output, gen_label_input])
        model = Model([gen_latent_input, gen_label_input], dis_prediction)
        self._compile_model(model)
        return model

    def get_model(self) -> Model:
        """Return the compiled GAN model."""
        return self.model

    def _compile_model(self, model: Model) -> None:
        """Compile the GAN model with the specified optimizer, loss, and metrics."""
        model.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
