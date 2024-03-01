import sys
import logging
import argparse
import yaml
import os

sys.path.append("src/")

from config import TRAIN_PARAMS
from dataloader import Loader
from generator import Generator
from discriminator import Discriminator
from trainer import Trainer
from test import Test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w",
    filename="./logs/cli-test.log",
)

params = dict()


def create_train_param(**kwargs):
    """
    Creates and saves training parameters to a YAML file.

    :param kwargs: A dictionary of training parameters.
    :type kwargs: dict

    **Keyword Arguments:**

    +-------------+------------------------------------------------------------------+----------+
    | Key         | Description                                                      | Type     |
    +=============+==================================================================+==========+
    | dataset     | Parameters related to the dataset including batch size,         | dict     |
    |             | image size, and normalization flag.                              |          |
    +-------------+------------------------------------------------------------------+----------+
    | train       | Training parameters including image size, number of epochs,      | dict     |
    |             | latent space dimensions, input channels, learning rate, beta1,   |          |
    |             | display flag, and device.                                        |          |
    +-------------+------------------------------------------------------------------+----------+
    """
    with open("./train_params.yml", "w") as f:
        yaml.safe_dump(
            {
                **kwargs["dataset"],
                **kwargs["train"],
            },
            f,
            default_flow_style=False,
        )


def cli():
    """
    CLI Tool for Image Model Training and Testing

    This command-line interface (CLI) application facilitates the training and testing of image models. It supports a range of options for configuring the dataset, model parameters, and execution mode.

    **Command-Line Arguments:**

    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | Argument       | Description                                                                    | Type    | Default |
    +================+================================================================================+=========+=========+
    | --dataset      | Path to the zip file containing the dataset.                                   | str     | None    |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --image_size   | Size of the images to be used.                                                 | int     | 64      |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --batch_size   | Number of images per training batch.                                            | int     | 64      |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --normalized   | Flag indicating if images should be normalized.                                 | bool    | True    |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --num_epochs   | The number of epochs for training.                                             | int     | 200     |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --latent_space | Dimensionality of the latent space.                                            | int     | 50      |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --in_channels  | The number of input channels for the model.                                    | int     | 1       |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --learning_rate| Learning rate for the optimizer.                                               | float   | 0.0002  |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --beta1        | The beta1 parameter for the Adam optimizer.                                    | float   | 0.5     |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --display      | Whether to display training progress and output images.                        | bool    | True    |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --device       | The device to run the model on ('cuda' or 'cpu').                              | str     | "cuda"  |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --train        | Flag to initiate training mode.                                                | action  | N/A     |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --test         | Flag to initiate testing mode.                                                 | action  | N/A     |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --model_path   | Path to the pre-trained model for testing.                                     | str     | None    |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --num_samples  | Number of samples to generate during testing.                                  | int     | 4       |
    +----------------+--------------------------------------------------------------------------------+---------+---------+
    | --label        | Label for the samples to be generated during testing.                          | int     | 0       |
    +----------------+--------------------------------------------------------------------------------+---------+---------+

    **Example Usage:**

    Training a model:
    ```
    python cli_script.py --train --dataset path/to/dataset.zip --image_size 64 --batch_size 64 --num_epochs 200 --latent_space 50 --in_channels 1 --learning_rate 0.0002 --beta1 0.5 --display True --device cuda
    ```

    Testing a model:
    ```
    python cli_script.py --test --model_path path/to/model.pth --num_samples 4 --label 0 --latent_space 50 --device cuda
    ```

    Note: Replace `cli_script.py` with the actual name of your Python script.
    """

    parser = argparse.ArgumentParser(description="Loading images".title())

    parser.add_argument("--dataset", type=str, help="path to the zip file".capitalize())
    parser.add_argument(
        "--image_size", type=int, default=64, help="size of the image".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size".capitalize()
    )
    parser.add_argument(
        "--normalized", type=bool, default=True, help="normalized".capitalize()
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
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to the model".capitalize()
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate".capitalize(),
    )
    parser.add_argument(
        "--label", type=int, default=0, help="Label for the samples".capitalize()
    )

    args = parser.parse_args()

    create_train_param(
        dataset={
            "dataset": {
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "normalized": args.normalized,
            }
        },
        train={
            "train": {
                "image_size": args.image_size,
                "num_epochs": args.num_epochs,
                "latent_space": args.latent_space,
                "in_channels": args.in_channels,
                "learning_rate": args.learning_rate,
                "beta1": args.beta1,
                "display": args.display,
                "device": args.device,
            }
        },
    )

    if args.dataset and args.train:
        logging.info("Creating dataloader".capitalize())

        if (
            args.image_size
            and args.batch_size
            and args.normalized
            and args.num_epochs
            and args.latent_space
            and args.in_channels
            and args.learning_rate
            and args.beta1
            and args.display
            and args.device
        ):
            loader = Loader(
                image_path=args.dataset,
                batch_size=args.batch_size,
                image_size=args.image_size,
                normalized=args.normalized,
            )
            logging.info("Extracting images".capitalize())
            loader.unzip_images()

            logging.info("Creating dataloader".capitalize())
            dataloader, labels = loader.create_dataloader()

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
            raise ValueError("Please provide all the arguments".capitalize())

    if args.test:
        test = Test(
            best_model_path=None if args.model_path is None else args.model_path,
            num_samples=args.num_samples,
            label=args.label,
            latent_space=args.latent_space,
            device=args.device,
        )

        test.test()
        test.create_gif_file()

        logging.info("Test complete.")


if __name__ == "__main__":
    cli()
