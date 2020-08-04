import numpy as np
import itertools

from typing import Tuple, \
                   Union, \
                   List
from keras import layers
from keras import activations
from keras import backend as K
from keras.layers import Layer


class ReverseEmbedding(Layer):
    def __init__(self, embedding_layer, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.activation = activations.get(activation)
        self.vocab_size = embedding_layer.get_config()['input_dim']
        self.trainable = False

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        assert len(inputs.shape) == 3, \
            'expected 3 dimensions'

        input_emb = inputs[:, -1, :]
        w_transpose = K.transpose(self.embedding_layer.embeddings)

        y = K.dot(input_emb, w_transpose)

        if self.activation is not None:
            y = self.activation(y)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_layer.embeddings.shape[0]


class RelativeEncoding(Layer):
    def __init__(self,
                 batch_size: int,
                 verbose: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.verbose = verbose
        self.sequence_length = None
        self.d_model = None
        self.encodings = None
        self.W_kr = None
        self.v = None

    def build(self,
              input_shape: Tuple):
        assert isinstance(input_shape, tuple), \
            f'received input_shape={input_shape}. Expected a tuple (i.e. single input).'
        assert len(input_shape) == 3, \
            f'expected shape with 3 dimensions: (batch_size, sequence_length, dimensions), ' \
            f'received shape with {len(input_shape)} dimensions: {input_shape}'
        self.sequence_length = input_shape[1]
        self.d_model = input_shape[2]

        self.W_kr = self.add_weight(name='W_k,r',
                                    shape=input_shape[1:],
                                    initializer='uniform',
                                    trainable=True)
        self.encodings = self.create_relative_encodings()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        y = self.encodings * self.W_kr

        if self.verbose:
            print(f'{self.__class__.__name__} call:')
            print(f'  encodings: {self.encodings.shape}')
            print(f'  W_kr:      {self.W_kr.shape}')
            print(f'  inputs:    {inputs.shape}')
            # print(f'  z:         {z.shape}')
            print(f'  y:         {y.shape}')

        assert len(inputs.shape) == len(y.shape), \
            f'unexpected length for produced output: ' \
            f'expected {inputs.shape}, ' \
            f'produced {y.shape}'
        assert inputs.shape[1:] == y.shape[1:], \
            f'unexpected shape for produced output: ' \
            f'expected {inputs.shape[1:]}, ' \
            f'produced {y.shape[1:]}'
        return y

    def compute_output_shape(self,
                             input_shape: Tuple):
        return input_shape

    def create_positional_encodings(self):
        embedding = [PE(pos, l, self.d_model) for pos, l in itertools.product(range(self.sequence_length),
                                                                              range(self.d_model))]
        embedding = np.array(embedding)
        embedding = embedding.reshape((self.sequence_length, self.d_model))
        return embedding

    def create_relative_encodings(self):
        embeddings = self.create_positional_encodings()
        embeddings = np.tile(embeddings, (self.batch_size, 1, 1))
        # embeddings = K.variable(embeddings)
        # embeddings._trainable = False
        return embeddings

    def relative_embedding(self,
                           i: int,
                           j: int):
        assert self.embeddings is not None, \
            'build the Positional Encoding layer before using it'
        delta = i - j
        delta = max(0, min(self.sequence_length, delta))

        return self.embeddings[delta]


def PE(pos, l, max_dimension):
    """Positional Embedding

    Arguments:
        pos: position in the sequence
        l: dimension, referred to in the paper as "i".
            Changed due to duplicated variable name
        max_dimension: maximum amount of dimensions used
    """
    alpha = pos/10000**(2*l/max_dimension)

    if l % 2 == 0:
        return np.sin(alpha)
    return np.cos(alpha)
