import sys
import logging
import argparse
import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/mimic.log",
)

sys.path.append("src/")

from utils import load_pickle
from config import to_save, MIMIC_PATH
from generator import Generator
from test import Test


class MimicSamples(Test):
    """
    This class generates mimic samples using a pre-trained AC-GAN model.

    :param model_path: Path to the saved model.
    :type model_path: str, optional
    :param latent_space: Dimensionality of the latent space.
    :type latent_space: int
    :param device: The device on which to perform computations.
    :type device: str
    :param num_samples: The number of samples to generate.
    :type num_samples: int

    **Methods:**

    .. list-table::
       :widths: 25 75
       :header-rows: 1

       * - Method
         - Description
       * - fetch_label
         - Retrieves a mapping of numerical labels to their corresponding class names.
       * - generate_nose_samples
         - Generates noise samples and their corresponding labels for image generation.
       * - save_generated_images
         - Saves generated images to the specified path.
       * - generate_mimic_samples
         - Generates and saves a specified number of mimic samples.

    **Examples:**

    .. code-block:: python

        # Initialize MimicSamples with default parameters
        mimic = MimicSamples(
            model_path="path/to/model.pth",
            latent_space=100,
            device="cuda",
            num_samples=1000
        )

        # Generate mimic samples
        mimic.generate_mimic_samples()
    """

    def __init__(
        self,
        model_path=None,
        latent_space=50,
        device="mps",
        num_samples=3000,
    ):
        super().__init__(model_path, latent_space, device)
        self.num_samples = num_samples

    def fetch_label(self):
        """
        Retrieves a dictionary mapping from numerical labels to class names.

        :return: A dictionary where keys are numerical labels and values are class names.
        :rtype: dict
        """
        if os.path.exists(to_save):
            labels = load_pickle(filename=os.path.join(to_save, "dataset.pkl"))
            return {value: key for key, value in labels.items()}
        else:
            raise Exception("No label found in the path".capitalize())

    def generate_nose_samples(self, label):
        """
        Generates noise samples and corresponding labels for image generation.

        :param label: The label for which to generate noise samples.
        :type label: int

        :return: A tuple of (noise samples tensor, labels tensor).
        :rtype: (torch.Tensor, torch.Tensor)
        """
        return torch.randn(self.num_samples, self.latent_space, 1, 1).to(
            self.device
        ), torch.full((self.num_samples,), label, dtype=torch.long).to(self.device)

    def save_generated_images(self, images=None, label_name=None):
        """
        Saves generated images to the specified directory.

        :param images: Tensor of images to be saved.
        :type images: torch.Tensor, optional
        :param label_name: Name of the label, used to create a subdirectory.
        :type label_name: str, optional
        """
        if images is not None and label_name is not None:
            if os.path.exists(MIMIC_PATH):
                label_path = os.path.join(MIMIC_PATH, label_name)
                os.makedirs(label_path, exist_ok=True)

                for index, image in enumerate(images):
                    image = image.detach().cpu().permute(1, 2, 0)
                    image = (image - image.min()) / (image.max() - image.min())
                    plt.imshow(image, cmap="gray")
                    plt.axis("off")
                    plt.savefig(
                        os.path.join(MIMIC_PATH, label_name, f"{index}.png"),
                        bbox_inches="tight",
                        pad_inches=0,
                    )
            else:
                raise Exception("Mimic path is not found".capitalize())

        else:
            raise ValueError("Images and label name are required".capitalize())

    def generate_mimic_samples(self):
        """
        Generates and saves a specified number of mimic samples using the AC-GAN model.
        """
        load_model = self.select_optimal_model()

        netG = Generator(
            latent_space=self.latent_space, num_labels=4, in_channels=1, image_size=64
        ).to(self.device)

        netG.load_state_dict(load_model)

        labels = self.fetch_label()

        for label, label_name in tqdm(labels.items()):
            noise_samples, specific_label = self.generate_nose_samples(label)
            mimic_images = netG(noise_samples, specific_label)

            self.save_generated_images(mimic_images, label_name)

            print(
                "{} is completed with number of mimic samples # {}\n".format(
                    label_name, self.num_samples
                )
            )

        print("Images saved in the folder - {}".format(MIMIC_PATH).title())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creating the mimic data using AC-GAN".title()
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the best model".capitalize(),
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=50,
        help="Size of the latent space".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run the code on".capitalize(),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3000,
        help="Number of samples to generate".capitalize(),
    )

    args = parser.parse_args()

    if args.num_samples and args.latent_space and args.device:
        logging.info("Generating {} samples".format(args.num_samples))

        mimic = MimicSamples(
            model_path=None if args.model_path is None else args.model_path,
            latent_space=args.latent_space,
            device=args.device,
            num_samples=args.num_samples,
        )
        logging.info("Generating Mimic Samples")

        mimic.generate_mimic_samples()

    else:
        raise Exception("Provide all the arguments first".capitalize())
