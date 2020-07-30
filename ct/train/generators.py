import numpy as np
from keras.utils import to_categorical


def next_token_batch_generator(*,
                               ct,
                               data=None,
                               data_path=None,
                               epochs=None,
                               epoch_steps=None,
                               batch_size,
                               sequence_length,
                               vocab_size):
    assert data is not None or data_path is not None, \
          'provide either a dataset or a path'
    if data is None:
        data = np.load(data_path)
    if epochs is None:
        epochs = int(1e16)
    if epoch_steps is None:
        epoch_steps = data.size[1] - data.size[1] % sequence_length
    if batch_size != sequence_length:
        raise NotImplementedError('each sample is a position in the sequence')

    def _next_token_batch_generator():
        for e in range(epochs):
            for i in range(0, epoch_steps - sequence_length - epoch_steps % sequence_length, batch_size):
                x_batch = [list(data[i + pos:i + pos + sequence_length]) for pos in range(batch_size)]
                x_batch = np.array(x_batch)
                x_batch = [x_batch, ct.memory, ct.compressed_memory]

                y_batch = [data[i + pos + sequence_length] for pos in range(batch_size)]
                y_batch = np.array(y_batch)
                y_batch = to_categorical(y_batch, num_classes=vocab_size)

                yield x_batch, y_batch
    return _next_token_batch_generator
