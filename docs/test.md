# GAN Model Tester Documentation

This document provides a comprehensive guide on how to use the `Test` class for generating and visualizing predictions from a trained Generator model in a GAN architecture.

## Overview

The `Test` class is designed to facilitate the evaluation of GAN models by generating images using a trained generator. It allows for the visualization of generated images and supports the creation of a GIF to showcase the model's performance over training epochs.

## Setup and Requirements

Ensure you have the following setup before proceeding:

- Python environment with required libraries: `torch`, `matplotlib`, `imageio`, and `numpy`.
- A trained Generator model saved as a `.pth` file.
- The `generator.py` script defining the Generator model class.
- `utils.py` script containing utility functions like `device_init` and `load_pickle`.

## Features

- **Model Testing**: Load a trained Generator model to generate images.
- **Visualization**: Visualize generated images with support for specifying the number of samples and labels for generation.
- **GIF Creation**: Create a GIF from images generated during training to visualize the model's learning progress.

## How to Use

### Running as a Standalone Script

Navigate to the directory containing the `test.py` script and run the following command in your terminal:

```bash
python test.py --model_path path/to/your/model.pth --num_samples 20 --label 1 --latent_space 50 --device cuda
```

## Parameters and Attributes

The `Test` class accepts several parameters at initialization:

| Parameter         | Type | Description                                      |
| ----------------- | ---- | ------------------------------------------------ |
| `best_model_path` | str  | Path to the best model's state dictionary.       |
| `num_samples`     | int  | Number of sample images to generate.             |
| `label`           | int  | Specific label for generating images.            |
| `latent_space`    | int  | Dimension of the latent space for the generator. |
| `device`          | str  | Computational device ('cuda', 'cpu', 'mps').     |

## Methods

- `__setup__`: Sets up the device for computation and loads necessary training parameters.
- `select_optimal_model`: Selects and loads the best-performing model based on saved checkpoints.
- `visualize_predictions`: Visualizes generated images.
- `generate_noise_samples`: Generates noise samples for the generator input.
- `test`: Main method for loading the model, generating images, and visualizing them.
- `create_gif_file`: Generates a GIF from images saved during training to visualize progress.

## Example Usage

```bash
python test.py --num_samples 4 --label 0 --latent_space 100 --device cuda
```

This command will load the best model (or the one specified), generate 4 images for label 0 with a latent space dimension of 100, and run the model on the CUDA device. Generated images will be visualized, and a GIF showing training progress will be created.

Ensure paths in `config.py` are correctly set for `BEST_MODELS`, `TEST_IMAGE_PATH`, and `TRAIN_GIF_PATH` to avoid path-related errors.
