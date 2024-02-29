import joblib as pkl
import os


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
