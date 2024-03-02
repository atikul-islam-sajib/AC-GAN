# CLI Tool for Mimic Data Generation Using AC-GAN

This document provides a comprehensive guide on using a command-line interface (CLI) tool designed for generating mimic data using a pre-trained AC-GAN model. The tool simplifies the process of synthetic data generation, allowing for the creation of large datasets that mimic the distribution of original data.

## Requirements

Before using the CLI tool, ensure you have the following:

- A Python environment with the necessary libraries installed (e.g., PyTorch, argparse).
- A pre-trained AC-GAN model saved as a `.pth` file.
- The `mimic.py` script, which contains the `MimicSamples` class for generating mimic samples.

## Features

- **Flexible Data Generation**: Generate thousands of synthetic images that mimic the original dataset's distribution.
- **Customizable Parameters**: Control the latent space dimension, computation device, and number of samples to generate through command-line arguments.
- **Ease of Use**: Simplify the data generation process with a CLI that requires only a few arguments to produce synthetic data.

## Command-Line Arguments

The CLI supports various arguments to control the model parameters and execution mode:

- `--model_path`: Path to the saved AC-GAN model. If not provided, attempts to use the best model based on saved checkpoints.
- `--latent_space`: Dimensionality of the latent space for the generator. Default is 50.
- `--device`: The computation device to use ('cuda', 'cpu', 'mps'). Default is 'mps'.
- `--num_samples`: The number of images to generate for each label in the dataset. Default is 3000.

## Usage

### Generating Mimic Samples

To generate mimic samples using a pre-trained AC-GAN model, execute the following command in your terminal:

```bash
python cli_mimic.py --model_path path/to/model.pth --latent_space 50 --device cuda --num_samples 3000
```

This command will generate 3000 samples for each label using the specified AC-GAN model, with computations performed on a CUDA device.

### Key Methods

- `generate_mimic_samples()`: The main method responsible for generating and saving mimic samples using the AC-GAN model. It loads the model, generates noise samples, and saves the generated images.

## Output

Generated images are saved in a directory specified by the `MIMIC_PATH` variable in the `config.py`. The tool creates a subdirectory for each class label, organizing the synthetic images accordingly.

## Additional Notes

- Ensure the `MIMIC_PATH` directory exists or is specified correctly in `config.py` to avoid errors during image saving.
- The `--device` argument allows for flexible usage across various hardware setups. Use 'cuda' for NVIDIA GPUs, 'cpu' for CPU-based generation, or 'mps' for Apple Silicon GPUs.
- The number of samples and latent space dimensions can be adjusted based on the needs of your dataset augmentation or synthetic data generation tasks.

By leveraging the capabilities of AC-GANs for synthetic data generation, the `MimicSamples` CLI tool provides an efficient solution for augmenting datasets, potentially enhancing machine learning model training and performance.
