# Mimic Data Generation Using AC-GAN

The `MimicSamples` class extends the `Test` class functionality to generate and save a large number of images mimicking the original dataset's distribution using a trained AC-GAN model. This tool is designed to augment datasets or create synthetic datasets for various applications, including but not limited to training machine learning models where data is scarce.

## Features

- **Label-specific Image Generation**: Generates images for specific labels in the dataset, allowing for targeted dataset augmentation.
- **High-volume Production**: Capable of generating thousands of samples to significantly augment your dataset.
- **Flexible Device Support**: Supports computation on different devices (CPU, CUDA, MPS) for efficient image generation.

## Prerequisites

- Python 3.x
- PyTorch
- Matplotlib
- tqdm
- A pre-trained AC-GAN model saved as a `.pth` file.
- The `utils.py` and `config.py` scripts for utility functions and configuration paths.

## Usage

### Command-Line Interface

The script provides a command-line interface (CLI) for easy use and integration into data processing pipelines.

#### Arguments

- `--model_path`: Path to the saved AC-GAN model. If not provided, the script attempts to use the best model based on saved checkpoints.
- `--latent_space`: The dimensionality of the latent space for the generator. Default is 50.
- `--device`: The computation device to use ('cuda', 'cpu', or 'mps'). Default is 'mps'.
- `--num_samples`: The number of images to generate for each label in the dataset. Default is 3000.

#### Example Command

```bash
python mimic_samples.py --model_path path/to/model.pth --latent_space 50 --device cuda --num_samples 3000
```

This command generates 3000 samples for each label using the specified AC-GAN model, with computations performed on a CUDA device.

### Key Methods

- `fetch_label()`: Retrieves a mapping of labels to class names from the dataset.
- `generate_nose_samples(label)`: Generates noise vectors and corresponding labels for image generation.
- `save_generated_images(images, label_name)`: Saves the generated images to a directory specific to each label.
- `generate_mimic_samples()`: Main method to generate and save mimic samples using the pre-trained AC-GAN model.

## Output

Generated images are saved in the directory specified by the `MIMIC_PATH` variable in `config.py`. Each class of images is saved in its subdirectory named after the class label.

## Example

After executing the script with the example command, the mimic data is structured as follows:

```
/mimic_data/
    /class_0/
        0.png
        1.png
        ...
    /class_1/
        0.png
        1.png
        ...
```

This structure facilitates easy integration of the generated data into existing datasets or use in training new models.

## Additional Notes

- Ensure the `MIMIC_PATH` directory exists or is correctly specified in `config.py`.
- Adjust the `--latent_space` and `--num_samples` according to the requirements of your specific use case.
- The class can be easily extended or modified for more customized data generation tasks.

By leveraging the capabilities of AC-GANs, the `MimicSamples` class provides a powerful tool for dataset augmentation, enabling the creation of diverse and balanced datasets for training robust machine learning models.
