import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/cli-mimic.log",
)

sys.path.append("src/")

from mimic import MimicSamples


def cli():
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


if __name__ == "__main__":
    cli()
