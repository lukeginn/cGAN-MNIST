from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
    Dropout,
    SpatialDropout2D,
    LeakyReLU,
    Reshape,
    Conv2DTranspose,
    Input,
    Embedding,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.datasets.mnist import load_data
import numpy as np

# from matplotlib import pyplot
import csv

# from tensorflow.keras.models import loadiscriminator_model

# TO RUN ON GPU ----
from tensorflow.python.keras import backend
from tensorflow.compat.v1 import ConfigProto, Session

sess = Session()
config = ConfigProto(log_device_placement=True)
sess = Session(config=config)
backend.set_session(sess)
# TO RUN ON GPU ----


def load_real_samples():
    # The first time you run this might be a bit slow, since the
    # mnist package has to download and cache the data.

    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Normalize the images
    train_images = train_images / 255
    test_images = test_images / 255

    # Transform the images for SimpleRNN
    # reshape(num_training_examples, num_timesteps, num_features)
    # timesteps and features is row length and number of columns respectively
    # you need to add the extra ,1 at the end for a CNN
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    return train_images, train_labels, test_images, test_labels


def generate_real_samples(images, labels, n_samples):
    ix = np.random.randint(0, images.shape[0], n_samples)
    X = images[ix]
    labels = labels[ix]
    # label smoothing
    y = np.ones((n_samples, 1)) * (0.9 + 0.1 * np.random.rand(n_samples, 1))

    return [X, labels], y


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    # generate random labels
    labels = np.random.randint(0, n_classes, n_samples)

    return [x_input, labels]


def generate_fake_samples(generator_model, latent_dim, n_samples):
    # generate points in latent space
    x_input, labels = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator_model.predict([x_input, labels])
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1)) + 0.1 * np.random.rand(n_samples, 1)

    return [X, labels], y


def save_plot(examples, epoch, acc_real, acc_fake, n=10):
    print("fix later")
    #    for i in range(n * n):
    #        pyplot.subplot(n, n, 1 + i)
    #        pyplot.axis('off')
    #        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')

    #    filename = 'C:/Users/Luke Ginn/Documents/Python Scripts/2021-09-07 - cGAN/Models/generated_plot_e%03d_acc_real_%.02d%%_acc_fake_%.02d%%.png' % (epoch+1, round(acc_real,2)*100, round(acc_fake,2)*100)
    #    pyplot.savefig(filename)
    #    pyplot.close()


def summarize_performance(
    epoch,
    generator_model,
    discriminator_model,
    g_loss,
    d_loss,
    images,
    labels,
    latent_dim,
    n_samples=100,
):

    [X_real, X_real_labels], y_real = generate_real_samples(images, labels, n_samples)
    _, acc_real = discriminator_model.evaluate(
        [X_real, X_real_labels], y_real, verbose=0
    )

    [X_fake, X_fake_labels], y_fake = generate_fake_samples(
        generator_model, latent_dim, n_samples
    )
    _, acc_fake = discriminator_model.evaluate(
        [X_fake, X_fake_labels], y_fake, verbose=0
    )

    # Printing Discriminator Performance
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

    # Get it to create the folder: Models if it does not exist

    # CSV File of Discriminator Performance
    with open(
        r"C:/Users/Luke Ginn/Documents/Python Scripts/2021-09-07 - cGAN/Models/Discriminator_Evaluation.csv",
        "a",
        newline="",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["Epoch", "Acc_Real", "Acc_Fake", "G_Loss", "D_Loss"]
        )
        writer.writerow(
            {
                "Epoch": epoch + 1,
                "Acc_Real": str(acc_real),
                "Acc_Fake": str(acc_fake),
                "G_Loss": g_loss,
                "D_Loss": d_loss,
            }
        )

    # Plot of Discriminator Performance
    save_plot(X_fake, epoch, acc_real, acc_fake)

    # Save GAN
    filename = (
        "C:/Users/Luke Ginn/Documents/Python Scripts/2021-09-07 - cGAN/Models/gan_model_%03d.h5"
        % (epoch + 1)
    )
    discriminator_model.trainable = False
    gan_model.save(filename)

    # Save Generator
    filename = (
        "C:/Users/Luke Ginn/Documents/Python Scripts/2021-09-07 - cGAN/Models/generator_model_%03d.h5"
        % (epoch + 1)
    )
    discriminator_model.trainable = True
    generator_model.save(filename)

    # Save Discriminator
    filename = (
        "C:/Users/Luke Ginn/Documents/Python Scripts/2021-09-07 - cGAN/Models/discriminator_model_%03d.h5"
        % (epoch + 1)
    )
    discriminator_model.trainable = True
    discriminator_model.save(filename)


def discriminator(n_classes=10):

    # Image Branch
    Image_Input = Input(shape=(28, 28, 1))

    # Label Branch
    Label_Input = Input(shape=(1,))
    Label_Branch = Embedding(n_classes, 50)(Label_Input)
    Label_Branch = Dense(28 * 28)(Label_Branch)
    Label_Branch = Reshape((28, 28, 1))(Label_Branch)

    # Merging Branches
    NeuralNet = Concatenate()([Image_Input, Label_Branch])
    NeuralNet = SpectralNormalization(Conv2D(128, (3, 3)))(NeuralNet)
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = MaxPooling2D((2, 2))(NeuralNet)
    NeuralNet = SpatialDropout2D(0.5)(NeuralNet)
    NeuralNet = SpectralNormalization(Conv2D(128, (3, 3)))(NeuralNet)
    NeuralNet = SpectralNormalization(Conv2D(128, (3, 3)))(NeuralNet)
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = MaxPooling2D((2, 2))(NeuralNet)
    NeuralNet = SpatialDropout2D(0.5)(NeuralNet)
    NeuralNet = Flatten()(NeuralNet)
    NeuralNet = SpectralNormalization(Dense(600))(NeuralNet)
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = Dropout(0.5)(NeuralNet)
    NeuralNet = SpectralNormalization(Dense(400))(NeuralNet)
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = Dropout(0.5)(NeuralNet)
    NeuralNet = SpectralNormalization(Dense(200))(NeuralNet)
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = Dropout(0.5)(NeuralNet)
    NeuralNet = SpectralNormalization(Dense(20))(NeuralNet)
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = Dropout(0.5)(NeuralNet)
    NeuralNet = SpectralNormalization(Dense(1, activation="sigmoid"))(NeuralNet)

    model = Model([Image_Input, Label_Input], NeuralNet)

    # model = loadiscriminator_model('C:/Users/Luke Ginn/Documents/Python Scripts/2021-09-07 - cGAN/Models/discriminator_model_065.h5')

    model.compile(
        optimizer=Adam(lr=0.0001, beta_1=0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.trainable = True

    return model


def generator(latent_dim, n_classes=10):

    # Latent Branch
    Latent_Input = Input(shape=(latent_dim,))
    Latent_Branch = SpectralNormalization(Dense(128 * 3 * 3))(Latent_Input)
    Latent_Branch = LeakyReLU(alpha=0.2)(Latent_Branch)
    Latent_Branch = BatchNormalization()(Latent_Branch)
    Latent_Branch = Reshape((3, 3, 128))(Latent_Branch)

    # Label Branch
    Label_Input = Input(shape=(1,))
    Label_Branch = Embedding(n_classes, 50)(Label_Input)
    Label_Branch = SpectralNormalization(Dense(3 * 3))(Label_Branch)
    Label_Branch = LeakyReLU(alpha=0.2)(Label_Branch)
    Label_Branch = BatchNormalization()(Label_Branch)
    Label_Branch = Reshape((3, 3, 1))(Label_Branch)

    # Merging Branches
    NeuralNet = Concatenate()([Latent_Branch, Label_Branch])
    NeuralNet = SpectralNormalization(Conv2DTranspose(128, (4, 4), strides=(1, 1)))(
        NeuralNet
    )
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = SpectralNormalization(Conv2DTranspose(128, (4, 4), strides=(2, 2)))(
        NeuralNet
    )
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = SpectralNormalization(
        Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same")
    )(NeuralNet)
    NeuralNet = LeakyReLU(alpha=0.2)(NeuralNet)
    NeuralNet = BatchNormalization()(NeuralNet)
    NeuralNet = SpectralNormalization(
        Conv2D(1, (10, 10), activation="sigmoid", padding="same")
    )(NeuralNet)

    model = Model([Latent_Input, Label_Input], NeuralNet)

    # model = loadiscriminator_model('C:/Users/Luke Ginn/Documents/Python Scripts/2021-09-07 - cGAN/Models/generator_model_065.h5')

    return model


def GAN(generator_model, discriminator_model):

    discriminator_model.trainable = False

    gen_Latent_Input, gen_Label_Input = generator_model.input
    gen_Image_Output = generator_model.output
    dis_prediction = discriminator_model([gen_Image_Output, gen_Label_Input])
    model = Model([gen_Latent_Input, gen_Label_Input], dis_prediction)

    # model = loadiscriminator_model('C:/Users/Luke Ginn/Documents/Python Scripts/2021-09-07 - cGAN/Models/gan_model_024.h5')

    model.compile(
        optimizer=Adam(lr=0.0002, beta_1=0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def GAN_Train(
    generator_model,
    discriminator_model,
    gan_model,
    images,
    labels,
    latent_dim,
    start_epoch=0,
    n_epochs=100,
    n_batch=256,
):

    bat_per_epo = int(images.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(start_epoch, n_epochs):
        for j in range(bat_per_epo):

            [X_real, X_real_labels], y_real = generate_real_samples(
                images, labels, half_batch
            )
            [X_fake, X_fake_labels], y_fake = generate_fake_samples(
                generator_model, latent_dim, half_batch
            )

            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            X_labels = np.concatenate((X_real_labels, X_fake_labels))

            d_loss, _ = discriminator_model.train_on_batch([X, X_labels], y)

            [X_gan, X_gan_labels] = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))

            g_loss, _ = gan_model.train_on_batch([X_gan, X_gan_labels], y_gan)

            print("Epoch: %d, Batch_No: %d/%d" % (i + 1, j + 1, bat_per_epo))

        # if (i+1) % 10 == 0:
        summarize_performance(
            i,
            generator_model,
            discriminator_model,
            g_loss,
            d_loss,
            images,
            labels,
            latent_dim,
        )


latent_dim = 100
discriminator_model = discriminator()
generator_model = generator(latent_dim)
gan_model = GAN(generator_model, discriminator_model)
train_images, train_labels, _, _ = load_real_samples()
GAN_Train(
    generator_model,
    discriminator_model,
    gan_model,
    train_images,
    train_labels,
    latent_dim,
    start_epoch=0,
)
# Turning off the GPU
sess.close()
