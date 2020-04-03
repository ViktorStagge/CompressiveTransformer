import numpy as np

from keras import layers
from keras import backend as K
from keras.layers import Layer

from model.utils import cosine_similarity


_default = dict(d_k=64,
                d_v=64,
                d_model=512,
                d_heads=2)


class ScaledDotProductAttention(Layer):
    def __init__(self, d_k=None, d_v=None, d_model=None, **kwargs):
        self.d_k = d_k or _default['d_k']
        self.d_v = d_v or _default['d_v']
        self.d_model = d_model or _default['d_model']
        self.w_q = None
        self.w_k = None
        self.w_v = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list), \
            f'received input_shape of type `{type(input_shape)}`, expected: `list`'
        shape_q, shape_k, shape_v = input_shape
        assert shape_q == (self.d_model, self.d_k) and \
               shape_k == (self.d_model, self.d_k) and \
               shape_v == (self.d_model, self.d_v), \
            f'unexpected input shapes received for: {input_shape}, expected:\n' \
            f'd_model= {self.d_model}\n' \
            f'd_k=d_q= {self.d_k}\n' \
            f'd_v    = {self.d_v}'

        self.w_q = self.add_weight(name='w_q',
                                   shape=(self.d_model, self.d_q),
                                   initializer='uniform',
                                   trainable=True)
        self.w_k = self.add_weight(name='w_k',
                                   shape=(self.d_model, self.d_k),
                                   initializer='uniform',
                                   trainable=True)
        self.w_v = self.add_weight(name='w_v',
                                   shape=(self.d_model, self.d_v),
                                   initializer='uniform',
                                   trainable=True)

        super().build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        q, k, v = x
        assert k.shape[0] == self.d_k and q.shape[0] == self.d_k and v.shape[0] == self.d_v, \
            'unexpected input shape received '  # (embedding_size, d_model)

        q = K.dot(q, self.w_q)
        k = K.dot(k, self.w_k)
        v = K.dot(v, self.w_v)

        z = K.dot(q, K.transpose(k))
        z = z / np.sqrt(self.d_k)  # optional Mask for decoder after this line
        z = K.softmax(z)
        z = K.dot(z, v)

        return z  # requires list (?). Probably no but disclaimer in docs.

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_q, shape_k, shape_v = input_shape

        return self.d_model, self.d_v


class MultiheadAttention(Layer):
    def __init__(self, d_heads=None, d_k=None, d_v=None, d_model=None, **kwargs):
        self.d_heads = d_heads or _default['d_heads']
        self.d_k = d_k or _default['d_k']
        self.d_v = d_v or _default['d_v']
        self.d_model = d_model or _default['d_model']

        self.heads = [ScaledDotProductAttention(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model) for _ in range(self.d_heads)]
        self.w_o = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list), \
            f'received input_shape of type `{type(input_shape)}`, expected: `list`'
        shape_q, shape_k, shape_v = input_shape
        assert shape_q == (self.d_model, self.d_k) and \
               shape_k == (self.d_model, self.d_k) and \
               shape_v == (self.d_model, self.d_v), \
            f'unexpected input shapes received for: {input_shape}, expected:\n' \
            f'd_model= {self.d_model}\n' \
            f'd_k=d_q= {self.d_k}\n' \
            f'd_v    = {self.d_v}'

        self.w_o = self.add_weight(name='w_o',
                                   shape=(self.d_heads*self.d_v, self.d_model),
                                   initializer='uniform',
                                   trainable=True)

        super().build(input_shape)

    def call(self, x, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_q, shape_k, shape_v = input_shape

        assert len(shape_v) == 2, f'receiving batch_size as well (?): {shape_v}'

        return self.shape_v[0], self.d_model


class ContentBasedAttention(Layer):
    def __init__(self, weight_layer=None, **kwargs):
        self.weight_layer = weight_layer
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list), 'expected to receive 2 tensors as input. Multiple inputs are not specified'
        assert len(x) == 2, f'expected to receive 2 tensors as input. Received {len(x)}'
        x1, x2 = x

        if self.weight_layer is not None:
            x1 = K.dot(x1, self.weight_layer.Q)
            x2 = K.dot(x2, self.weight_layer.K)
            x3 = K.dot(x2, self.weight_layer.V)

        y = cosine_similarity(x1, x2)

        if self.weight_layer is not None:
            y = K.dot(y, x3)

        return y

    def compute_output_shape(self, input_shape):
        pass
