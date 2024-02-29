import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/generator.log",
)

sys.path.append("src/")

from utils import total_params


class Generator(nn.Module):
    """
    A generator module for generating images from a latent space and class labels.

    This class defines a generator architecture for a GAN model, which takes a noise vector
    and class labels as input and produces an image.

    **Methods**

    +------------------+-------------------------------------------------------------------+
    | Method           | Description                                                       |
    +==================+===================================================================+
    | connected_layer  | Constructs the generator network layers based on configuration.   |
    +------------------+-------------------------------------------------------------------+
    | forward          | Defines the forward pass of the generator.                        |
    +------------------+-------------------------------------------------------------------+

    **Parameters**

    +--------------+--------+---------------------------------------------------------------+
    | Parameter    | Type   | Description                                                   |
    +==============+========+===============================================================+
    | latent_space | int    | Dimension of the latent space.                                |
    +--------------+--------+---------------------------------------------------------------+
    | num_labels   | int    | Number of unique labels for conditional generation.           |
    +--------------+--------+---------------------------------------------------------------+
    | image_size   | int    | Size of the generated images (height and width).              |
    +--------------+--------+---------------------------------------------------------------+
    | in_channels  | int    | Number of channels in the generated images.                   |
    +--------------+--------+---------------------------------------------------------------+
    """

    def __init__(self, latent_space=50, num_labels=4, image_size=64, in_channels=1):
        self.latent_space = latent_space
        self.num_labels = num_labels
        self.image_size = image_size
        self.in_channels = in_channels

        super(Generator, self).__init__()
        self.config_layers = [
            (self.latent_space * 2, self.image_size * 8, 4, 1, 0, True, False),
            (self.image_size * 8, self.image_size * 4, 4, 2, 1, True, False),
            (self.image_size * 4, self.image_size * 2, 4, 2, 1, True, False),
            (self.image_size * 2, self.image_size, 4, 2, 1, True, False),
            (self.image_size, self.in_channels, 4, 2, 1, False, False),
        ]
        self.labels = nn.Embedding(
            num_embeddings=self.num_labels, embedding_dim=self.latent_space
        )
        self.model = self.connected_layer(config_layers=self.config_layers)

    def connected_layer(self, config_layers=None):
        """
        Constructs the generator network layers from the provided configuration.

        **Parameters**

        +---------------+---------------------------+-----------------------------------------+
        | Parameter     | Type                      | Description                             |
        +===============+===========================+=========================================+
        | config_layers | list of tuples            | Configuration for each layer in the    |
        |               |                           | generator network.                      |
        +---------------+---------------------------+-----------------------------------------+

        **Returns**

        A sequential model comprising all the configured layers.
        """
        if config_layers is not None:
            layers = OrderedDict()

            for idx, (
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                batch_norm,
                bias,
            ) in enumerate(config_layers[:-1]):
                layers[f"ConvTranspose{idx+1}"] = nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                )

                if batch_norm:
                    layers[f"BatchNorm{idx+1}"] = nn.BatchNorm2d(out_channels)

                layers[f"ReLU{idx+1}"] = nn.ReLU(inplace=True)

            (
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                batch_norm,
                bias,
            ) = config_layers[-1]
            layers[f"outConvTranspose"] = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )
            layers["outLayer"] = nn.Tanh()

            return nn.Sequential(layers)

        else:
            raise ValueError("config layer should be defined".capitalize())

    def forward(self, noise, labels):
        """
        Defines the forward pass of the generator.

        **Parameters**

        +----------+------------+--------------------------------------------+
        | Parameter| Type       | Description                                |
        +==========+============+============================================+
        | noise    | torch.Tensor| The latent space noise vector.             |
        +----------+------------+--------------------------------------------+
        | labels   | torch.Tensor| The class labels for conditional generation.|
        +----------+------------+--------------------------------------------+

        **Returns**

        A generated image tensor.
        """
        labels = self.labels(labels)
        labels = labels.view(labels.size(0), self.latent_space, 1, 1)
        return self.model(torch.cat((noise, labels), dim=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator model".capitalize())
    parser.add_argument(
        "--latent_dim", type=int, default=50, help="latent dimension".capitalize()
    )
    parser.add_argument(
        "--num_labels", type=int, default=4, help="number of labels".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="image size".capitalize()
    )
    parser.add_argument(
        "--in_channels", type=int, default=1, help="number of channels".capitalize()
    )
    parser.add_argument("--netG", action="store_true", help="netG".capitalize())

    args = parser.parse_args()

    if args.netG:
        if args.latent_dim and args.num_labels and args.image_size and args.in_channels:
            net_G = Generator(
                latent_space=args.latent_dim,
                num_labels=args.num_labels,
                image_size=args.image_size,
                in_channels=args.in_channels,
            )

            logging.info(f"Generator Model: {net_G}")
            logging.info(
                "Total params in Generator # {}".format(total_params(model=net_G))
            )

        else:
            raise ValueError("Please provide all the required arguments".capitalize())

    else:
        raise ValueError("Please provide all the required arguments".capitalize())
