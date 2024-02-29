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
    filename="discriminator.log",
)

sys.path.append("src/")

from utils import total_params


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_labels=4, image_size=64):
        self.in_channels = in_channels
        self.num_labels = num_labels
        self.image_size = image_size

        super(Discriminator, self).__init__()

        self.config_layers = [
            (self.in_channels, self.image_size, 4, 2, 1, False),
            (self.image_size, self.image_size * 2, 4, 2, 1, True),
            (self.image_size * 2, self.image_size * 4, 4, 2, 1, True),
            (self.image_size * 4, self.image_size * 8, 4, 2, 1, True),
            (self.image_size * 8, 1 + self.num_labels, 4, 1, 0),
        ]

        self.model = self.connected_layer(config_layers=self.config_layers)

    def connected_layer(self, config_layers=None):
        if config_layers is not None:
            layers = OrderedDict()

            for idx, (
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                batch_norm,
            ) in enumerate(config_layers[:-1]):
                layers["conv{}".format(idx + 1)] = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
                if batch_norm:
                    layers[f"batchNorm{idx+1}"] = nn.BatchNorm2d(
                        num_features=out_channels
                    )

                layers[f"leaky_relu{idx+1}"] = nn.LeakyReLU(
                    negative_slope=0.2, inplace=True
                )

            (in_channels, out_channels, kernel_size, stride, padding) = config_layers[
                -1
            ]
            layers["out"] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

            return nn.Sequential(layers)

        else:
            raise ValueError("config layer should be defined".capitalize())

    def forward(self, x):
        output = self.model(x)
        real_or_fake = output[:, 0:1]
        labels = output[:, 1:]
        return torch.sigmoid(real_or_fake.view(-1, 1)), F.log_softmax(
            labels.view(-1, self.num_labels)
        )


if __name__ == "__main__":

    net_D = Discriminator()

    noise_data = torch.randn(64, 1, 64, 64)
    real_fake, labels = net_D(noise_data)
    print(real_fake.shape, labels.shape)
    print(total_params(net_D))
