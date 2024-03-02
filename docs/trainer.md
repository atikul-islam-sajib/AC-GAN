# AC-GAN Model Trainer

The AC-GAN Model Trainer is a comprehensive training framework designed to facilitate the efficient training of Auxiliary Classifier Generative Adversarial Networks (AC-GANs). This framework includes setup, execution, and monitoring of the training process, leveraging the capabilities of PyTorch for deep learning tasks.

## Features

- **Flexible Training Configuration**: Customize various training parameters including image size, epochs, latent space dimensionality, and learning rates.
- **Dual Model Architecture**: Simultaneously trains a Generator and a Discriminator model, incorporating class labels into the generative process.
- **Progress Monitoring**: Provides options for real-time progress display or logging, including loss metrics and epoch details.
- **Checkpoint Saving**: Automatically saves model checkpoints and generated image samples at specified intervals.
- **Device Compatibility**: Supports training on different devices including CPUs, GPUs, and Apple's Metal Performance Shaders (MPS).

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Matplotlib (optional, for image visualization)

## Installation

Ensure you have PyTorch installed along with the other required packages. Installation instructions for PyTorch can be found at [the official PyTorch website](https://pytorch.org/get-started/locally/).

## Usage

The trainer can be executed from the command line with various options to customize the training process:

### Command Line Arguments

| Argument          | Description                                    | Default Value  |
| ----------------- | ---------------------------------------------- | -------------- |
| `--image_size`    | Size of the images to generate (square).       | 64             |
| `--num_epochs`    | Total number of training epochs.               | 200            |
| `--latent_space`  | Dimensionality of the latent space.            | 50             |
| `--in_channels`   | Number of input channels (e.g., 3 for RGB).    | 1              |
| `--learning_rate` | Learning rate for the Adam optimizer.          | 0.0002         |
| `--beta1`         | Beta1 hyperparameter for Adam optimizer.       | 0.5            |
| `--display`       | Display training progress on stdout.           | True           |
| `--device`        | Device to run the training on (`cuda`, `cpu`). | `cuda`         |
| `--train`         | Flag to initiate the training process.         | Not applicable |

### Example Command

```shell
python trainer.py --train --image_size 64 --num_epochs 200 --latent_space 100 --in_channels 3 --learning_rate 0.0002 --beta1 0.5 --display True --device cuda
```

This command initiates the training of an AC-GAN model with 64x64 RGB images, using a 100-dimensional latent space, for 200 epochs on a CUDA-enabled device.

## Customization

- **Model Parameters**: Adapt the generator and discriminator architectures within their respective classes to experiment with different GAN configurations.
- **Training Loop**: Modify the training loop in the `Trainer` class to include additional logging, metrics, or custom behavior during the training process.
- **Data Preparation**: Use the `load_dataloader` utility to customize the preprocessing and loading of your dataset for training.

## Output

- **Model Checkpoints**: Saved models during training, located in the directory specified by `TRAIN_MODELS` and `BEST_MODELS` in `config.py`.
- **Generated Images**: Sample images generated during training are saved at intervals defined in the training parameters, facilitating visual progress tracking.
- **Loss Metrics**: Training loss for both the generator and discriminator is logged, allowing for performance evaluation over time.
