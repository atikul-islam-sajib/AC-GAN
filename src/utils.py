import joblib as pkl
import os
import torch
import torch.nn as nn
from config import to_save
import yaml


def pickle(value=None, filename=None):
    if value and filename:
        pkl.dump(value=value, filename=filename)
    else:
        raise ValueError("value and filename are required".capitalize())


def load_pickle(filename=None):
    if filename is not None:
        return pkl.load(filename=filename)


def clean(path=None):
    if path:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    else:
        raise ValueError("path is required".capitalize())


def total_params(model=None):
    return sum(p.numel() for p in model.parameters())


def weight_init(m):

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def load_dataloader():
    if os.path.exists(to_save):
        dataloader_path = os.path.join(to_save, "dataloader.pkl")
        try:
            return load_pickle(filename=dataloader_path)

        except Exception as e:
            print("Exception caught in the section - {}".format(e))
    else:
        raise Exception("Dataloader is not found in the processed folder".capitalize())


def train_parameters():
    with open("./default_train.yml", "r") as file:
        return yaml.safe_load(file)
