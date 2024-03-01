import sys
import logging
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w",
    filename="./logs/trainer.log",
)

sys.path.append("src/")

from utils import (
    weight_init as weight,
    device_init,
    load_dataloader,
    train_parameters as params,
    clean,
)
from config import TRAIN_MODELS, BEST_MODELS, TRAIN_IMAGE_PATH
from generator import Generator
from discriminator import Discriminator


class Trainer:
    """
    Trainer class for training GAN models including a Generator and a Discriminator.

    This class manages the training process of a GAN model, including setup, training loops,
    saving checkpoints, and displaying training progress.

    **Methods**

    +---------------------+-------------------------------------------------------------------+
    | Method              | Description                                                       |
    +=====================+===================================================================+
    | __setup__           | Initializes training setup including models, optimizers, and loss |
    |                     | functions.                                                        |
    +---------------------+-------------------------------------------------------------------+
    | train_discriminator | Trains the discriminator model with real and fake images.         |
    +---------------------+-------------------------------------------------------------------+
    | train_generator     | Trains the generator model to fool the discriminator.             |
    +---------------------+-------------------------------------------------------------------+
    | save_checkpoints    | Saves model checkpoints during training.                          |
    +---------------------+-------------------------------------------------------------------+
    | show_progress       | Displays or logs training progress.                               |
    +---------------------+-------------------------------------------------------------------+
    | train               | Executes the training process for the specified number of epochs. |
    +---------------------+-------------------------------------------------------------------+

    **Parameters**

    +----------------+--------+---------------------------------------------------------------+
    | Parameter      | Type   | Description                                                   |
    +================+========+===============================================================+
    | image_size     | int    | Size of the images for the GAN model.                         |
    +----------------+--------+---------------------------------------------------------------+
    | num_epochs     | int    | Number of epochs for training.                                |
    +----------------+--------+---------------------------------------------------------------+
    | latent_space   | int    | Size of the latent space for the generator.                   |
    +----------------+--------+---------------------------------------------------------------+
    | num_labels     | int    | Number of labels for conditional GANs.                        |
    +----------------+--------+---------------------------------------------------------------+
    | in_channels    | int    | Number of input channels for images (e.g., 3 for RGB).        |
    +----------------+--------+---------------------------------------------------------------+
    | learning_rate  | float  | Learning rate for the optimizer.                              |
    +----------------+--------+---------------------------------------------------------------+
    | beta1          | float  | Beta1 hyperparameter for Adam optimizer.                      |
    +----------------+--------+---------------------------------------------------------------+
    | display        | bool   | Flag to display progress to stdout instead of logging.        |
    +----------------+--------+---------------------------------------------------------------+
    | num_steps      | int    | Interval of steps to show/log progress.                       |
    +----------------+--------+---------------------------------------------------------------+
    | device         | str    | Device to run the training on ('cuda', 'cpu', 'mps').         |
    +----------------+--------+---------------------------------------------------------------+
    """

    def __init__(
        self,
        image_size=64,
        num_epochs=200,
        latent_space=50,
        num_labels=4,
        in_channels=1,
        learning_rate=0.0002,
        beta1=0.5,
        display=True,
        num_steps=50,
        device="mps",
    ):
        self.image_size = image_size
        self.num_epochs = num_epochs
        self.latent_space = latent_space
        self.num_labels = num_labels
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.is_display = display
        self.num_steps = num_steps
        self.device = device

        self.__setup__()

    def __setup__(self):
        """
        Sets up the training environment, including the device, data loader, model initialization, and optimizers.
        """
        self.device = device_init(device=self.device)
        self.dataloader = load_dataloader()
        self.params = params()
        clean(path=TRAIN_MODELS)
        clean(path=BEST_MODELS)

        self.netG = Generator(
            latent_space=self.latent_space,
            num_labels=self.num_labels,
            image_size=self.image_size,
            in_channels=self.in_channels,
        ).to(self.device)

        self.netD = Discriminator(
            in_channels=self.in_channels,
            num_labels=self.num_labels,
            image_size=self.image_size,
        ).to(self.device)

        self.netG.apply(weight)
        self.netD.apply(weight)

        self.optimizerG = optim.Adam(
            params=self.netG.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.params["train"]["beta2"]),
        )
        self.optimizerD = optim.Adam(
            params=self.netD.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.params["train"]["beta2"]),
        )

        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        self.G_loss = list()
        self.D_loss = list()
        self.G_D_loss_epoch = {"G_loss": list(), "D_loss": list()}
        self.loss = float("inf")

    def train_discriminator(self, **kwargs):
        """
        Trains the discriminator model using both real and fake images.

        **Parameters**

        +------------------+----------------+-----------------------------------------------+
        | Parameter        | Type           | Description                                 |
        +==================+================+===============================================+
        | real_images      | torch.Tensor   | Real images from the dataset.                |
        +------------------+----------------+-----------------------------------------------+
        | labels           | torch.Tensor   | Real labels corresponding to the real images.|
        +------------------+----------------+-----------------------------------------------+
        | fake_samples     | torch.Tensor   | Fake images generated by the generator.      |
        +------------------+----------------+-----------------------------------------------+
        | real_labels      | torch.Tensor   | Labels indicating real images.               |
        +------------------+----------------+-----------------------------------------------+
        | fake_labels      | torch.Tensor   | Labels indicating fake images.               |
        +------------------+----------------+-----------------------------------------------+
        | generated_labels | torch.Tensor   | Labels generated by the generator for        |
        |                  |                | conditional generation.                      |
        +------------------+----------------+-----------------------------------------------+

        **Returns**

        Total loss for the discriminator as a float.
        """
        self.optimizerD.zero_grad()

        real_predict, real_aux_labels = self.netD(kwargs["real_images"])
        fake_predict, fake_aux_labels = self.netD(kwargs["fake_samples"])

        total_D_loss = self.params["train"]["factor"] * (
            self.adversarial_loss(real_predict, kwargs["real_labels"])
            + self.auxiliary_loss(real_aux_labels, kwargs["labels"])
        ) + self.params["train"]["factor"] * (
            self.adversarial_loss(fake_predict, kwargs["fake_labels"])
            + self.auxiliary_loss(fake_aux_labels, kwargs["generated_labels"])
        )

        total_D_loss.backward(retain_graph=True)
        self.optimizerD.step()

        return total_D_loss.item()

    def train_generator(self, **kwargs):
        """
        Trains the generator model to generate images that can fool the discriminator.

        **Parameters are similar to `train_discriminator` method.**

        **Returns**

        Total loss for the generator as a float.
        """
        self.optimizerG.zero_grad()

        fake_predict, fake_aux_labels = self.netD(kwargs["fake_samples"])

        total_G_loss = self.params["train"]["factor"] * self.adversarial_loss(
            fake_predict, kwargs["real_labels"]
        ) + self.params["train"]["factor"] * self.auxiliary_loss(
            fake_aux_labels, kwargs["generated_labels"]
        )

        total_G_loss.backward(retain_graph=True)
        self.optimizerG.step()

        return total_G_loss.item()

    def save_checkpoints(self, **kwargs):
        """
        Saves checkpoints of the generator model during training.

        **Parameters**

        +-----------+--------+---------------------------------------------+
        | Parameter | Type   | Description                                 |
        +===========+========+=============================================+
        | epoch     | int    | Current epoch number.                       |
        +-----------+--------+---------------------------------------------+
        | g_loss    | float  | Current generator loss.                     |
        +-----------+--------+---------------------------------------------+
        """
        torch.save(
            self.netG.state_dict(),
            os.path.join(TRAIN_MODELS, "netG_{}.pth".format(kwargs["epoch"])),
        )
        if kwargs["epoch"] > self.params["train"]["num_steps"]:
            if kwargs["g_loss"] < self.loss:
                self.loss = kwargs["g_loss"]

                torch.save(
                    {
                        "G_load_state_dict": self.netG.state_dict(),
                        "loss": kwargs["g_loss"],
                        "epochs": kwargs["epoch"],
                    },
                    os.path.join(BEST_MODELS, "netG_{}.pth".format(kwargs["epoch"])),
                )

    def show_progress(self, **kwargs):
        """
        Displays or logs the training progress.

        **Parameters**

        +-----------+--------+-------------------------------------------------+
        | Parameter | Type   | Description                                     |
        +===========+========+=================================================+
        | epoch     | int    | Current epoch number.                           |
        +-----------+--------+-------------------------------------------------+
        | num_epochs| int    | Total number of epochs for training.            |
        +-----------+--------+-------------------------------------------------+
        | index     | int    | Current batch index within the epoch.           |
        +-----------+--------+-------------------------------------------------+
        | len_dataloader | int | Total number of batches in the dataloader.    |
        +-----------+--------+-------------------------------------------------+
        | d_loss    | float  | Current discriminator loss.                     |
        +-----------+--------+-------------------------------------------------+
        | g_loss    | float  | Current generator loss.                         |
        +-----------+--------+-------------------------------------------------+
        """
        if self.is_display:
            print(
                "[Epochs - {}/{}] [Steps - {}/{}] [D loss: {}] [G loss: {}]".format(
                    kwargs["epoch"],
                    kwargs["num_epochs"],
                    kwargs["index"],
                    kwargs["len_dataloader"],
                    kwargs["d_loss"],
                    kwargs["g_loss"],
                ),
                "\n",
            )
        else:
            logging.info(
                "[Epochs - {}/{}] [Steps - {}/{}] [D loss: {}] [G loss: {}]".format(
                    kwargs["epoch"],
                    kwargs["num_epochs"],
                    kwargs["index"],
                    kwargs["len_dataloader"],
                    kwargs["d_loss"],
                    kwargs["g_loss"],
                )
            )

    def save_epoch_images(self, **kwargs):
        """
        Saves a grid of generated images at specified epochs during training.

        This method generates a set of images using the current state of the generator model and saves them as a single image file, allowing for visual inspection of the model's progress over time.

        **Parameters**

        +----------------+------------------+-----------------------------------------------------------+
        | Parameter      | Type             | Description                                               |
        +================+==================+===========================================================+
        | epoch          | int              | The current epoch number. Used to determine if images     |
        |                |                  | should be saved based on the interval specified in        |
        |                |                  | training parameters.                                      |
        +----------------+------------------+-----------------------------------------------------------+
        | noise_samples  | torch.Tensor     | A batch of noise samples to feed into the generator       |
        |                |                  | for image generation.                                     |
        +----------------+------------------+-----------------------------------------------------------+
        | generated_labels | torch.Tensor   | The labels associated with the noise samples, used for    |
        |                  |                | conditional image generation.                             |
        +----------------+------------------+-----------------------------------------------------------+

        **Behavior**

        - The method checks if the current epoch meets the saving criteria based on the interval specified in `params["train"]["num_steps"]`.
        - If the condition is met, it generates images using the generator model and saves them in the TRAIN_IMAGE_PATH directory with a filename indicating the epoch number.
        - Raises an exception if the TRAIN_IMAGE_PATH does not exist.

        **Exceptions**

        +------------------+-----------------------------------------------------------------------+
        | Exception        | Condition                                                            |
        +==================+=======================================================================+
        | Exception        | Raised if the TRAIN_IMAGE_PATH directory does not exist.              |
        +------------------+-----------------------------------------------------------------------+
        """
        if os.path.exists(TRAIN_IMAGE_PATH):

            if kwargs["epoch"] % self.params["train"]["num_steps"] == 0:
                images = self.netG(kwargs["noise_samples"], kwargs["generated_labels"])

                image_path = os.path.join(
                    TRAIN_IMAGE_PATH, "image_{}.png".format(str(kwargs["epoch"]))
                )
                save_image(
                    images,
                    image_path,
                    nrow=8,
                    normalize=True,
                )
        else:
            raise Exception("TRAIN_IMAGE_PATH does not exist".capitalize())

    def train(self):
        """
        Executes the training loop for the specified number of epochs.
        """
        for epoch in range(self.num_epochs):
            for index, (real_images, labels) in enumerate(self.dataloader):

                real_images = real_images.to(self.device)
                labels = labels.to(self.device)

                batch_size = real_images.shape[0]
                noise_samples = torch.randn(
                    batch_size,
                    self.latent_space,
                    self.params["train"]["noise_channel"],
                    self.params["train"]["noise_channel"],
                ).to(self.device)
                real_labels = torch.ones(
                    batch_size, self.params["train"]["noise_channel"]
                ).to(self.device)
                fake_labels = torch.zeros(
                    batch_size, self.params["train"]["noise_channel"]
                ).to(self.device)
                generated_labels = torch.randint(0, self.num_labels, (batch_size,)).to(
                    self.device
                )

                fake_samples = self.netG(noise_samples, generated_labels)

                d_loss = self.train_discriminator(
                    real_images=real_images,
                    labels=labels,
                    fake_samples=fake_samples,
                    real_labels=real_labels,
                    fake_labels=fake_labels,
                    generated_labels=generated_labels,
                )

                g_loss = self.train_generator(
                    fake_samples=fake_samples,
                    real_labels=real_labels,
                    generated_labels=generated_labels,
                )

                if index % self.num_steps == 0:
                    self.show_progress(
                        epoch=epoch + 1,
                        num_epochs=self.num_epochs,
                        index=index,
                        len_dataloader=len(self.dataloader),
                        d_loss=d_loss,
                        g_loss=g_loss,
                    )

                    self.D_loss.append(d_loss)
                    self.G_loss.append(g_loss)

            self.show_progress(
                epoch=epoch + 1,
                num_epochs=self.num_epochs,
                index=len(self.dataloader),
                len_dataloader=len(self.dataloader),
                d_loss=np.mean(self.D_loss),
                g_loss=np.mean(self.G_loss),
            )

            self.save_checkpoints(epoch=epoch + 1, g_loss=np.mean(self.G_loss))

            self.G_D_loss_epoch["G_loss"].append(np.mean(self.G_loss))
            self.G_D_loss_epoch["D_loss"].append(np.mean(self.D_loss))

            self.save_epoch_images(
                noise_samples=noise_samples,
                epoch=epoch + 1,
                generated_labels=generated_labels,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the AC-GAN model".title())
    parser.add_argument(
        "--image_size", type=int, default=64, help="The size of the image".capitalize()
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="The number of epochs".capitalize()
    )
    parser.add_argument(
        "--latent_space", type=int, default=50, help="The latent space".capitalize()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="The number of input channels".capitalize(),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0002,
        help="The learning rate".capitalize(),
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="The beta1 parameter".capitalize()
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=True,
        help="Whether to display the progress".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to run the model on".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:
        if (
            args.image_size
            and args.num_epochs
            and args.latent_space
            and args.in_channels
            and args.learning_rate
            and args.beta1
            and args.display
            and args.device
        ):

            trainer = Trainer(
                image_size=args.image_size,
                num_epochs=args.num_epochs,
                latent_space=args.latent_space,
                in_channels=args.in_channels,
                learning_rate=args.learning_rate,
                beta1=args.beta1,
                display=args.display,
                device=args.device,
            )
            trainer.train()
        else:
            raise ValueError("Define all the arguments please".capitalize())
    else:
        raise ValueError("Train the model first".capitalize())
