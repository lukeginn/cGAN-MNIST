# cGAN-MNIST

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/yourusername/cGAN-MNIST/releases)

This repository contains the code and configuration files for the cGAN-MNIST project. The project aims to generate MNIST digit images using a Conditional Generative Adversarial Network (cGAN).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Introduction

The cGAN-MNIST project addresses the challenge of generating realistic MNIST digit images conditioned on class labels. By leveraging a Conditional Generative Adversarial Network (cGAN), this project builds a model that can generate digit images corresponding to specific digit classes.

This project is structured into several key stages:

1. **Data Loading and Preprocessing**: Loading and preprocessing the MNIST dataset.
2. **Model Definition**: Defining the Generator and Discriminator models.
3. **Model Compilation**: Compiling the GAN model.
4. **Model Training**: Training the cGAN model on the MNIST dataset.
5. **Model Evaluation**: Evaluating the performance of the trained cGAN model.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Configuration

The configuration file `config.yaml` contains various settings and hyperparameters for the project. You can find this file in the root directory of the repository. Modify this file to adjust the behavior of the cGAN model.

## Usage

Run the main script:

```bash
python main.py
```

## Code Overview

- `main.py`: The main script to train the cGAN model.
- `check_gpu.py`: Utility to check GPU availability.
- `data_reader.py`: Module to read the MNIST dataset.
- `data_processor.py`: Module to process the MNIST dataset.
- `sample_processor.py`: Module to process samples.
- `performance_processor.py`: Module to evaluate model performance.
- `discriminator.py`: Defines the Discriminator model.
- `generator.py`: Defines the Generator model.
- `gan_compiler.py`: Compiles the GAN model.
- `gan_trainer.py`: Trains the GAN model.

## License

This project is licensed under the MIT License.