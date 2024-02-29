import sys
import os
import torch
import unittest

sys.path.append("src/")

from utils import load_pickle
from config import to_save


class TestImageFolder(unittest.TestCase):
    def setUp(self):
        self.dataloader = load_pickle(os.path.join(to_save, "dataloader.pkl"))
        self.dataset = load_pickle(os.path.join(to_save, "dataset.pkl"))

    def test_quantity_dataset(self):
        self.assertEqual(len(self.dataloader.dataset), 6400)

    def test_num_labels(self):
        self.assertEqual(len(self.dataset), 4)

    def test_shape_dataset(self):
        data, label = next(iter(self.dataloader))
        self.assertEqual(data.shape, torch.Size([64, 1, 64, 64]))


if __name__ == "__main__":
    unittest.main()
