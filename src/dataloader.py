import sys
import logging
import argparse
import os
import zipfile
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/dataloader.log",
)

sys.path.append("src/")

from utils import pickle
from config import to_extract, to_save


class Loader:
    """
    A loader class for processing image datasets.

    The Loader class is designed to handle the unzipping of image datasets, their
    normalization, and the creation of a DataLoader for batch processing.

    **Methods**

    +-------------------+-----------------------------------------------------------------+
    | Method            | Description                                                     |
    +===================+=================================================================+
    | unzip_images      | Extracts images from a zip file to a specified directory.       |
    +-------------------+-----------------------------------------------------------------+
    | _normalized       | Returns a torchvision transform for normalizing images.         |
    +-------------------+-----------------------------------------------------------------+
    | class_to_idx      | Provides a mapping from class names to indices.                 |
    +-------------------+-----------------------------------------------------------------+
    | create_dataloader | Creates a DataLoader for the unzipped and optionally normalized |
    |                   | image dataset.                                                  |
    +-------------------+-----------------------------------------------------------------+

    **Parameters**

    +----------------+-----------------------------------+----------------------------------------------------+
    | Parameter      | Type                              | Description                                        |
    +================+===================================+====================================================+
    | image_path     | str                               | Path to the zip file containing the image dataset. |
    +----------------+-----------------------------------+----------------------------------------------------+
    | batch_size     | int                               | Batch size for the DataLoader.                     |
    +----------------+-----------------------------------+----------------------------------------------------+
    | image_size     | int                               | The size (height, width) images are resized to.    |
    +----------------+-----------------------------------+----------------------------------------------------+
    | normalized     | bool                              | Whether to normalize images.                       |
    +----------------+-----------------------------------+----------------------------------------------------+
    """

    def __init__(self, image_path=None, batch_size=64, image_size=64, normalized=True):
        self.image_path = image_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_normalized = normalized

    def unzip_images(self):
        """
        Extracts the dataset from a zip file.

        **Exceptions**

        +----------------+----------------------------------------------------------------+
        | Exception      | Condition                                                      |
        +================+================================================================+
        | Exception      | Raised if the specified extraction path does not exist.       |
        +----------------+----------------------------------------------------------------+
        """
        with zipfile.ZipFile(self.image_path, "r") as zip_ref:
            if os.path.exists(to_extract):
                zip_ref.extractall(to_extract)
            else:
                raise Exception("Extracting images failed".capitalize())

    def _normalized(self):
        """
        Defines the normalization transform for images.

        **Returns**

        A torchvision.transforms.Compose object with the normalization and resizing transforms.
        """
        if self.use_normalized:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 5, 0.5]),
                    transforms.Grayscale(num_output_channels=1),
                ]
            )

            return transform

    @staticmethod
    def class_to_idx(dataset=None):
        """
        Maps class names to indices.

        **Parameters**

        +-----------+---------------------------+-----------------------------------------+
        | Parameter | Type                      | Description                             |
        +===========+===========================+=========================================+
        | dataset   | torchvision.datasets      | The dataset to map class names from.    |
        +-----------+---------------------------+-----------------------------------------+

        **Returns**

        A dictionary mapping class names to indices.
        """
        if dataset is not None:
            return dataset.class_to_idx

    def create_dataloader(self):
        """
        Creates and saves a DataLoader and class-to-index mapping.

        **Exceptions**

        +----------------+----------------------------------------------------------------+
        | Exception      | Condition                                                      |
        +================+================================================================+
        | Exception      | Raised if the specified paths for extraction or saving do not |
        |                | exist.                                                         |
        +----------------+----------------------------------------------------------------+

        **Returns**

        A tuple containing the DataLoader and a dictionary mapping class names to indices.
        """
        if os.path.exists(to_extract):
            datasets = ImageFolder(
                root=os.path.join(to_extract, "Dataset"), transform=self._normalized()
            )
            dataloader = DataLoader(datasets, batch_size=self.batch_size, shuffle=True)

            if os.path.exists(to_save):

                try:
                    pickle(
                        value=dataloader,
                        filename=os.path.join(to_save, "dataloader.pkl"),
                    )
                    pickle(
                        value=Loader.class_to_idx(dataset=datasets),
                        filename=os.path.join(to_save, "dataset.pkl"),
                    )
                except Exception as e:
                    print(e)
            else:
                raise Exception("Creating dataloader failed".capitalize())
        else:
            raise Exception(
                "Extracting images failed from the create dataloader method".capitalize()
            )

        return dataloader, datasets.class_to_idx


if __name__ == "__main__":
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

    args = parser.parse_args()

    if args.dataset:
        logging.info("Creating dataloader".capitalize())

        if args.image_size and args.batch_size and args.normalized:
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
        else:
            raise ValueError("Please provide all the arguments".capitalize())
    else:
        raise ValueError("Please provide the path to the zip file".capitalize())
