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
        assert q.shape[0] == self.d_k and q.shape[1] == self.d_model, \
            f'unexpected input shape received for `Q: Query`: {q.shape}'  # (embedding_size, d_model)
        assert k.shape[0] == self.d_k and k.shape[1] == self.d_model, \
            f'unexpected input shape received for `K: Key`: {k.shape}'  # (embedding_size, d_model)
        assert v.shape[0] == self.d_v and v.shape[1] == self.d_model, \
            f'unexpected input shape received for `V: Value`: {v.shape}'  # (embedding_size, d_model)

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

        assert len(shape_v) == 2, f'receiving batch_size as well (?): {shape_v}'

        return self.d_model, self.d_v


class MultiheadAttention(Layer):
    def __init__(self, d_heads=None, d_k=None, d_v=None, d_model=None, **kwargs):
        self.d_heads = d_heads or _default['d_heads']
        self.d_k = d_k or _default['d_k']
        self.d_v = d_v or _default['d_v']
        self.d_model = d_model or _default['d_model']

        # self.heads = [ScaledDotProductAttention(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model) for _ in range(self.d_heads)]
        self.w_o = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list), \
            f'received input_shape of type `{type(input_shape)}`, expected: `list`'
        assert len(input_shape) == self.d_heads, \
            f'unexpected input length. Expected {self.d_heads}. Received {len(input_shape)}'
        assert all(shape == input_shape[0] for shape in input_shape), \
            f'received input tensors of varying shapes'

        self.w_o = self.add_weight(name='w_o',
                                   shape=(self.d_heads*self.d_v, self.d_model),
                                   initializer='uniform',
                                   trainable=True)

        super().build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list)

        z = K.concatenate(tensors=x)
        y = K.dot(z, self.w_o)
        return y

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape_q, shape_k, shape_v = input_shape

        # return self.shape_q[0], self.d_model  # ([batch_size, ]shape_q[0], d_model
        shape = input_shape[-1]

        return shape[0], self.d_v


class ContentBasedAttention(Layer):
    """Content Based Attention as defined in "Neural Turing Machines" by Graves et. al.
    """
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

        z = cosine_similarity(x1, x2)
        y = K.softmax(z)

        if self.weight_layer is not None:
            y = K.dot(y, x3)

        return y

    def compute_output_shape(self, input_shape):
        return input_shape[:-1], None


class ContentBasedAttention_CT(Layer):
    """Content Based Attention as defined in the Compressive Transformer paper.
    def attn(h, m) <- sigma((hQ)(mK))(mV)  # most likely softmax(•)
    """
    def __init__(self, weight_layer=None, **kwargs):
        self.weight_layer = weight_layer
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list), \
            'expected to receive 2 tensors as input. Multiple inputs are not specified'
        assert len(x) == 2, \
            f'expected to receive 2 tensors as input. Received {len(x)}'
        assert self.weight_layer is not None
        h, m = x

        hQ = K.dot(h, self.weight_layer.Q)
        mK = K.dot(m, self.weight_layer.K)
        mV = K.dot(m, self.weight_layer.V)

        z = K.dot(hQ, mK)
        z = K.softmax(z)
        y = K.dot(z, mV)

        return y

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2, \
            'expected to receive 2 tensors as input. Multiple inputs are not specified'
        assert len(input_shape) == 2, \
            f'expected to receive 2 tensors as input. Received {len(input_shape)}'
        shape_h, shape_m = input_shape

        return shape_h[0], shape_m[1]
