import sys
import logging
import argparse
import os
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/mimic.log",
)

sys.path.append("src/")

from utils import device_init, load_dataloader, params, clean, load_pickle
from config import TRAIN_MODELS, BEST_MODELS, to_save
from generator import Generator
from discriminator import Discriminator
from trainer import Trainer


class MimicSamples:
    def __init__(
        self,
        model_path=None,
        latent_space=50,
        num_samples=3000,
        image_size=64,
        device="mps",
    ):
        self.model_path = model_path
        self.latent_space = latent_space
        self.num_samples = num_samples
        self.device = device
        self.image_size = image_size

        self.initialize_device()

    def initialize_device(self):
        device_init(device=self.device)

    def load_best_model(self):
        if self.model_path is None:
            if os.path.exists(BEST_MODELS):
                model_files = os.listdir(BEST_MODELS)
                max_epoch = max(
                    torch.load(os.path.join(BEST_MODELS, model_file))["epochs"]
                    for model_file in model_files
                )

                best_model_filename = "netG_{}.pth".format(max_epoch)
                best_model_path = os.path.join(BEST_MODELS, best_model_filename)

                best_model = torch.load(best_model_path)
                G_load_state_dict = best_model["G_load_state_dict"]

                return G_load_state_dict
            else:
                raise Exception("No model found in the path".capitalize())
        else:
            return torch.load(self.model_path)["G_state_dict"]

    def fetch_label(self):
        if os.path.exists(to_save):
            labels = load_pickle(filename=os.path.join(to_save, "dataset.pkl"))
            return {value: key for key, value in labels.items()}
        else:
            raise Exception("No label found in the path".capitalize())

    def produce_noise_and_labels(self, label):
        return torch.randn(self.num_samples, self.latent_space, 1, 1).to(
            self.device
        ), torch.full((self.num_samples), label, dtype=torch.long).to(self.device)

    def generate_mimic_samples(self):
        load_model = self.load_best_model()

        netG = Generator(
            latent_space=self.latent_space,
            num_labels=0,
            in_channels=1,
        )

        netG.load_state_dict(load_model).to(self.device)

        labels = self.fetch_label()

        for label, label_name in labels.items():
            noise_samples, specific_label = self.produce_noise_and_labels(label)
            mimic_images = netG(noise_samples, specific_label)

            print(mimic_images.shape, label_name, label)


if __name__ == "__main__":

    mimic = MimicSamples(device=None, latent_space=50, num_samples=100, image_size=64)
