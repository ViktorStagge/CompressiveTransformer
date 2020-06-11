import numpy as np
from keras.utils import to_categorical


def next_token_batch_generator(*,
                               ct,
                               data=None,
                               data_path=None,
                               epochs,
                               epoch_steps,
                               sequence_length,
                               vocab_size):
    assert data is not None or data_path is not None, \
          'provide either a dataset or a path'
    if data is None:
        data = np.load(data_path)

    def _next_token_batch_generator():
        for e in range(epochs):
            for i in range(0, epoch_steps - epoch_steps % sequence_length, sequence_length):
                x_batch = data[:, i:i + sequence_length]
                x_batch = [x_batch, ct.memory, ct.compressed_memory]

                y_batch = data[:, i + sequence_length]
                y_batch = to_categorical(y_batch, num_classes=vocab_size)

                yield x_batch, y_batch
    return _next_token_batch_generator
