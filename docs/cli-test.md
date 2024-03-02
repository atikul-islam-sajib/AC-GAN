# CLI Tool for GAN Model Training and Testing

This document provides an overview and guide on using a command-line interface (CLI) tool designed for training and testing Generative Adversarial Networks (GANs) on image data. The tool simplifies the process of configuring the training environment, executing the training process, and testing the trained models by generating sample images.

## Setup Requirements

Before using the CLI tool, ensure you have the following setup:

- Python environment with necessary libraries installed (e.g., PyTorch, argparse, yaml).
- The dataset is prepared in a zip file format for training.
- The project directory includes the necessary Python scripts (`generator.py`, `discriminator.py`, `dataloader.py`, etc.) and a `config.py` file for paths and parameters configuration.

## Features

- **Data Loading**: Automate the process of loading and preprocessing image datasets.
- **Model Training**: Train GAN models with customizable parameters via command-line arguments.
- **Model Testing**: Test trained models by generating images and optionally creating a GIF to visualize the training progression.
- **Parameter Configuration**: Define training parameters dynamically and save them to a YAML file for easy sharing and reproducibility.

## Command-Line Arguments

The CLI tool supports various arguments to control the dataset, training parameters, and execution mode:

- `--dataset`: Path to the dataset zip file.
- `--image_size`: Image resolution to be used during training.
- `--batch_size`: Number of images per training batch.
- `--normalized`: Flag indicating whether to normalize images.
- `--num_epochs`: Total number of training epochs.
- `--latent_space`: Dimensionality of the latent space vector.
- `--in_channels`: Number of input channels (e.g., 3 for RGB images).
- `--learning_rate`: Optimizer learning rate.
- `--beta1`: Beta1 hyperparameter for the Adam optimizer.
- `--display`: Flag to enable progress display.
- `--device`: Computational device ('cuda', 'cpu', or 'mps').
- `--train`: Flag to initiate training mode.
- `--test`: Flag to initiate testing mode.
- `--model_path`: Path to a pre-trained model (for testing).
- `--num_samples`: Number of samples to generate during testing.
- `--label`: Label for generated samples during testing.

## Example Usage

### Training a Model

```sh
python cli_script.py --train --dataset path/to/dataset.zip --image_size 64 --batch_size 64 --num_epochs 200 --latent_space 50 --in_channels 1 --learning_rate 0.0002 --beta1 0.5 --display True --device cuda
```

This command initiates the training process with the specified dataset and parameters.

### Testing a Model

```sh
python cli_script.py --test --model_path path/to/model.pth --num_samples 4 --label 0 --latent_space 50 --device cuda
```

This command generates images using a trained model specified by `--model_path`.

## Implementation Details

The CLI tool orchestrates the training and testing process by:

- Dynamically creating and saving training parameters to a YAML file.
- Loading and preprocessing the dataset.
- Initializing the Generator and Discriminator models.
- Training the models according to the specified parameters.
- Testing the trained models by generating images.
- Optionally, creating a GIF from training images to visualize progression.

## Additional Notes

- Ensure the `config.py` file correctly specifies paths for saving models, logs, and generated images.
- The `--display` flag allows for real-time progress monitoring in the console. Disable it if running in environments without a display (e.g., remote servers).
- Adjust the `--device` argument according to your hardware capabilities to utilize GPU acceleration if available.

This CLI tool streamlines the process of working with GANs for image generation tasks, making it accessible to users with varying levels of expertise in machine learning and deep learning.
