import numpy as np

from keras import layers
from keras import backend as K
from keras.layers import Layer
from omegaconf import OmegaConf

from model.utils import cosine_similarity


class ReverseEmbedding(Layer):
    def __init__(self, embedding_layer, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.activation = activation
        self.vocab_size = embedding_layer.get_config()['input_dim']
        self.trainable = False

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        assert len(inputs.shape) == 3, \
            'expected 3 dimensions'

        input_emb = inputs[:, -1, :]
        w_transpose = K.transpose(self.embedding_layer.embeddings)  # OK with w_tranpose?

        print(f'input_emb.shape={input_emb.shape}')
        print(f'w_tanspose.shape={w_transpose.shape}')

        y = K.dot(input_emb, w_transpose)
        if self.activation == 'softmax':
            y = K.softmax(y, axis=-1)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_layer.embeddings.shape[0]
