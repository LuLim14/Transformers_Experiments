import os

import torch

from ImdbHandler import ImdbDataset


if __name__ == "__main__":
    imdb_ds = ImdbDataset()
    print(imdb_ds.train_dataset.shape, imdb_ds.eval_dataset.shape)
