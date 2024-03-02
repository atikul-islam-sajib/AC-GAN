# DataLoader for Image Datasets

This DataLoader utility is designed to simplify the handling of image datasets. It encompasses functionalities for extracting images from a zip file, normalizing them, and batching them for processing. Below is a detailed explanation of its capabilities and how to use it.

## Features

- **Image Extraction**: Unzip your image dataset into a specified directory for processing.
- **Normalization and Resizing**: Apply normalization and resizing to the images in your dataset.
- **Batch Processing**: Easily create a `DataLoader` for batch processing of your images.
- **Class Index Mapping**: Generate a mapping from class names to indices, useful for training models.

## Requirements

To use this DataLoader, you need to have Python installed along with the following packages:

- `torch`
- `torchvision`
- `zipfile`
- `os`

## Setup

Before running the script, ensure you have a configuration file (`config.py`) with the following variables:

- `to_extract`: Path where the zip file will be extracted.
- `to_save`: Path where the DataLoader and class-to-index mapping will be saved.

Additionally, a utility module (`utils.py`) is required for pickling the DataLoader and class-to-index mapping.

## Usage

The DataLoader can be used via the command line with the following arguments:

| Argument       | Type | Description                             | Default |
| -------------- | ---- | --------------------------------------- | ------- |
| `--dataset`    | str  | Path to the zip file containing images. |         |
| `--image_size` | int  | Size (height, width) to resize images.  | 64      |
| `--batch_size` | int  | Number of images per batch.             | 64      |
| `--normalized` | bool | Apply normalization to images.          | True    |

### Example Command

```sh
python your_script_name.py --dataset path/to/your/dataset.zip --image_size 128 --batch_size 32 --normalized True
```

This command will:

1. Extract images from `dataset.zip`.
2. Normalize and resize the images to 128x128 pixels.
3. Create a DataLoader with a batch size of 32.
4. Save the DataLoader and class-to-index mapping to the specified `to_save` directory.

## Output

- A DataLoader object for the processed image dataset.
- A pickle file (`dataloader.pkl`) containing the DataLoader.
- A pickle file (`dataset.pkl`) containing the class-to-index mapping.
