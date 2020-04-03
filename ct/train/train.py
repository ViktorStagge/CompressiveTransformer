import numpy as np

from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.optimizers import Adam

import load
from model import model as create_model
from preprocess import preprocess


def train(**kwargs):
    dataset = load.treebank(path_to_dir='../data/input/treebank/treebank/raw/')
    processed = preprocess(dataset)
    model = create_model()

    print(dataset.head(3))
    print(processed.head(3))
    print(model.summary())

