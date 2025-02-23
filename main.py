from src.utils.check_gpu import check_gpu
from src.data.data_reader import DataReader
from src.data.data_processor import DataProcessor
from src.model.sample_processor import SampleProcessor
from src.model.performance_processor import PerformanceProcessor
from src.model.discriminator import Discriminator
from src.model.generator import Generator
from src.model.gan_compiler import GANCompiler
from src.model.gan_trainer import GANTrainer

physical_devices = check_gpu()

data_reader = DataReader()
data_processor = DataProcessor()
sample_processor = SampleProcessor()
performance_processor = PerformanceProcessor(sample_processor, "outputs")
discriminator = Discriminator()
generator = Generator(latent_dim=100)

data = data_reader.load_data()
train_images, train_labels, test_images, test_labels = data_processor.run(data)

discriminator.build_model()
discriminator_model = discriminator.get_model()

generator.build_model()
generator_model = generator.get_model()

gan_compiler = GANCompiler(generator_model, discriminator_model)
gan_compiler.build_model()
gan_model = gan_compiler.get_model()

gan_trainer = GANTrainer(
    generator_model,
    discriminator_model,
    gan_model,
    train_images,
    train_labels,
    100,
    sample_processor,
    performance_processor,
)
gan_trainer.train_model()
generator_model, discriminator_model, gan_model = gan_trainer.get_models()
