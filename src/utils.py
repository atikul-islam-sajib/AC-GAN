import joblib as pkl
import os


def pickle(value=None, filename=None):
    if value and filename:
        pkl.dump(value=value, filename=filename)
    else:
        raise ValueError("value and filename are required".capitalize())


def clean(path=None):
    if path:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    else:
        raise ValueError("path is required".capitalize())
