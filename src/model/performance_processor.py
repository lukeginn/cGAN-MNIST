import os
import numpy as np
import csv
from keras.models import Model
from dataclasses import dataclass
from typing import Tuple
from src.model.sample_processor import SampleProcessor


@dataclass
class PerformanceProcessor:
    sample_processor: SampleProcessor
    save_dir: str
    latent_dim: int = 100

    def summarize_performance(
        self,
        epoch: int,
        generator_model: Model,
        discriminator_model: Model,
        g_loss: float,
        d_loss: float,
        images: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 100,
    ) -> None:
        """Summarize the performance of the GAN model."""
        [X_real, X_real_labels], y_real = self.sample_processor.real_samples(
            images, labels, n_samples
        )
        _, acc_real = discriminator_model.evaluate(
            [X_real, X_real_labels], y_real, verbose=0
        )

        [X_fake, X_fake_labels], y_fake = self.sample_processor.fake_samples(
            generator_model, self.latent_dim, n_samples
        )
        _, acc_fake = discriminator_model.evaluate(
            [X_fake, X_fake_labels], y_fake, verbose=0
        )

        self.print_performance(epoch, acc_real, acc_fake, g_loss, d_loss)
        self.save_performance_to_csv(epoch, acc_real, acc_fake, g_loss, d_loss)
        self.save_plot(X_fake, epoch, acc_real, acc_fake)
        self.save_models(epoch, generator_model, discriminator_model)

    def print_performance(
        self, epoch: int, acc_real: float, acc_fake: float, g_loss: float, d_loss: float
    ) -> None:
        """Print the performance of the discriminator."""
        print(
            "epoch: %03d, acc_real: %.02d%%, acc_fake: %.02d%%, d=%.3f, g=%.3f"
            % (
                epoch + 1,
                round(acc_real, 2) * 100,
                round(acc_fake, 2) * 100,
                g_loss,
                d_loss,
            )
        )

    def save_performance_to_csv(
        self, epoch: int, acc_real: float, acc_fake: float, g_loss: float, d_loss: float
    ) -> None:
        """Save the performance of the discriminator to a CSV file."""
        csv_path = f"{self.save_dir}/Discriminator_Evaluation.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["Epoch", "Acc_Real", "Acc_Fake", "G_Loss", "D_Loss"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "Epoch": epoch + 1,
                    "Acc_Real": str(acc_real),
                    "Acc_Fake": str(acc_fake),
                    "G_Loss": g_loss,
                    "D_Loss": d_loss,
                }
            )

    def save_plot(
        self,
        examples: np.ndarray,
        epoch: int,
        acc_real: float,
        acc_fake: float,
        n: int = 10,
    ) -> None:
        """Save a plot of the generated images."""
        print("fix later")
        #    for i in range(n * n):
        #        pyplot.subplot(n, n, 1 + i)
        #        pyplot.axis('off')
        #        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')

        #    filename = f'{self.save_dir}/generated_plot_e{epoch+1:03d}_acc_real_{round(acc_real,2)*100:.02d}%_acc_fake_{round(acc_fake,2)*100:.02d}%.png'
        #    pyplot.savefig(filename)
        #    pyplot.close()

    def save_models(
        self, epoch: int, generator_model: Model, discriminator_model: Model
    ) -> None:
        """Save the GAN, generator, and discriminator models."""
        gan_model_path = f"{self.save_dir}/gan_model_{epoch+1:03d}.h5"
        generator_model_path = f"{self.save_dir}/generator_model_{epoch+1:03d}.h5"
        discriminator_model_path = (
            f"{self.save_dir}/discriminator_model_{epoch+1:03d}.h5"
        )

        discriminator_model.trainable = False
        generator_model.save(gan_model_path)

        discriminator_model.trainable = True
        generator_model.save(generator_model_path)
        discriminator_model.save(discriminator_model_path)
