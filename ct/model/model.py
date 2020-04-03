from keras import activations
from keras import regularizers
from keras import callbacks
from keras.models import Model
from keras.models import Sequential as SequentialModel
from keras.layers import Embedding, LSTM, Dense, Dropout


def model():
    model = naive_model()
    return model


def naive_model():
    model = SequentialModel()
    model.add(Embedding(input_dim=1000, output_dim=100))
    model.add(LSTM(units=64))
    model.add(Dense(100))
    model.add(Dropout(rate=0.2))
    model.add(Dense(50))

    model.compile(optimizer='Adam',
                  loss='mse')
    return model
