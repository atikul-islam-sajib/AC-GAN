# Generator Module for GAN

The Generator module is a core component designed for generating synthetic images from a latent space and class labels, forming an integral part of Generative Adversarial Networks (GANs). This document provides an overview of its functionalities, parameters, and usage.

## Features

- **Image Generation**: Generates images by mapping from a latent space and conditioned on class labels.
- **Configurable Architecture**: Allows customization of the network structure to suit various image generation tasks.
- **Conditional Generation**: Supports conditional image generation using class labels.

## Requirements

To utilize this module, ensure you have Python installed along with the following packages:

- `torch`
- `torch.nn`
- `torch.nn.functional`
- `collections.OrderedDict`

Python and these libraries should be installed and accessible in your environment.

## Parameters and Attributes

### Constructor Parameters

| Parameter      | Type | Description                                         |
| -------------- | ---- | --------------------------------------------------- |
| `latent_space` | int  | Dimensionality of the latent space.                 |
| `num_labels`   | int  | Number of unique labels for conditional generation. |
| `image_size`   | int  | The height and width of the generated images.       |
| `in_channels`  | int  | Number of output channels in the generated images.  |

### Attributes

- `config_layers`: Defines the configuration for each layer in the generator network.
- `labels`: An embedding layer for handling class labels.
- `model`: The sequential model comprising all configured layers, constructed in `connected_layer`.

### Methods

- **`connected_layer`**: Constructs the network layers based on provided configurations.
- **`forward`**: Performs a forward pass through the network, generating an image from noise and class labels.

## Usage Example

### Command Line

```sh
python generator.py --latent_dim 100 --num_labels 10 --image_size 64 --in_channels 3 --netG
```

This command initializes the Generator with a 100-dimensional latent space, 10 unique labels, generates 64x64 images with 3 channels (RGB).

### Code Snippet

```python
latent_dim = 100
num_labels = 10
image_size = 64
in_channels = 3

net_G = Generator(
    latent_space=latent_dim,
    num_labels=num_labels,
    image_size=image_size,
    in_channels=in_channels,
)

noise = torch.randn((1, latent_dim, 1, 1))
labels = torch.randint(0, num_labels, (1,))

generated_image = net_G(noise, labels)
```

This example demonstrates how to initialize the Generator, create a noise vector, select a class label, and generate an image.

## Additional Information

- Ensure to configure the logging and utility paths correctly to access the `total_params` function and logging functionalities.
- The network's architecture can be customized via the `config_layers` attribute to experiment with different generator designs.
