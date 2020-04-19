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
        print(f'#### INIT [{self.__class__.__name__}] ####')
        self.d_k = d_k or _default['d_k']
        self.d_v = d_v or _default['d_v']
        self.d_model = d_model or _default['d_model']
        self.w_q = None
        self.w_k = None
        self.w_v = None

        self.d_q = self.d_k

        super().__init__(**kwargs)

    def build(self, input_shape):
        print(f'#### BUILD [{self.__class__.__name__}] ####')
        # print(input_shape)
        if isinstance(input_shape, list):
            assert len(input_shape) in (2, 3), \
                f'received input_shape of length `{len(input_shape)}`, expected length in (2, 3)'
            if len(input_shape) == 3:
                shape_q, shape_k, shape_v = input_shape
            elif len(input_shape) == 2:
                shape_q, shape_k = input_shape
                shape_v = shape_k
        else:
            assert isinstance(input_shape, tuple)
            shape_q = input_shape
            shape_k = input_shape
            shape_v = input_shape

        assert shape_q == (None, None, self.d_model) and \
               shape_k == (None, None, self.d_model) and \
               shape_v == (None, None, self.d_model) \
               or \
               shape_q == (None, self.d_model, self.d_model) and \
               shape_k == (None, self.d_model, self.d_model) and \
               shape_v == (None, self.d_model, self.d_model), \
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
        print(f'#### CALL [{self.__class__.__name__}] ####')
        if isinstance(x, list):
            if len(x) == 3:
                q, k, v = x
            elif len(x) == 2:
                q, k = x
                v = k
        else:
            q = x
            k = x
            v = x

        assert [dim.value for dim in q.shape.dims] == [None, None, self.d_model] \
            or [dim.value for dim in q.shape.dims] == [None, self.d_model, self.d_model], \
            f'unexpected input shape received for `Q: Query`: {q.shape}'
        assert [dim.value for dim in k.shape.dims] == [None, None, self.d_model]\
            or [dim.value for dim in k.shape.dims] == [None, self.d_model, self.d_model], \
            f'unexpected input shape received for `K: Key`: {k.shape}'
        assert [dim.value for dim in v.shape.dims] == [None, None, self.d_model]\
            or [dim.value for dim in v.shape.dims] == [None, self.d_model, self.d_model], \
            f'unexpected input shape received for `V: Value`: {v.shape}'

        q = K.dot(q, self.w_q)
        k = K.dot(k, self.w_k)
        v = K.dot(v, self.w_v)

        k_T = K.permute_dimensions(k, pattern=(0, 2, 1))
        z = K.batch_dot(q, k_T)
        z = z / np.sqrt(self.d_k)  # optional Mask for decoder after this line
        z = K.softmax(z)
        y = K.batch_dot(z, v)

        print(f'    q={q.shape},\n'
              f'    k={k.shape},\n'
              f'    v={v.shape}\n'
              f'    k_T={k_T.shape}\n'
              f'    y={y.shape}')

        return y

    def compute_output_shape(self, input_shape):
        print('#### COMPUTE OUTPUT SHAPE ####')
        # assert isinstance(input_shape, list)
        # assert len(input_shape) == 3
        # assert len(input_shape[-1]) == 3
        # shape_q, shape_k, shape_v = input_shape

        print((None, self.d_model, self.d_v))

        return None, self.d_model, self.d_v


class MultiHeadAttention(Layer):
    def __init__(self, d_heads=None, d_k=None, d_v=None, d_model=None, **kwargs):
        self.d_heads = d_heads or _default['d_heads']
        self.d_k = d_k or _default['d_k']
        self.d_v = d_v or _default['d_v']
        self.d_model = d_model or _default['d_model']

        self.w_o = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list), \
            f'received input_shape of type `{type(input_shape)}`, expected: `list`'
        assert len(input_shape) == self.d_heads, \
            f'unexpected input length. Expected {self.d_heads}. Received {len(input_shape)}'
        assert all(shape == input_shape[0] for shape in input_shape), \
            f'received varying input shapes'

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
        head_shape = input_shape[-1]

        return None, self.d_model, self.d_model  # None, head_shape[1], self.d_model


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
        return input_shape


def content_based_attention(h, m, w_q, w_k, w_v):
    if isinstance(w_q, list) or isinstance(w_k, list) or isinstance(w_v, list):
        raise NotImplementedError('multiple heads will be implemented soon. For now pass one head per layer.')

    hQ = K.dot(h, self.weight_layer.w_k)
    mK = K.dot(m, self.weight_layer.w_k)
    mV = K.dot(m, self.weight_layer.w_v)

    z = K.batch_dot(hQ, mK)
    z = K.softmax(z)
    y = K.batch_dot(z, mV)

    return y


class ContentBasedAttention_CT(Layer):
    """Content Based Attention as defined in the Compressive Transformer paper:
    `def attn(h, m) <- sigma((hQ)(mK))(mV)`.  # most likely softmax(•)
    """
    def __init__(self, heads=None, **kwargs):
        self.heads = heads
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

        hQ = K.dot(h, self.weight_layer.w_k)
        mK = K.dot(m, self.weight_layer.w_k)
        mV = K.dot(m, self.weight_layer.w_v)

        z = K.batch_dot(hQ, mK)
        z = K.softmax(z)
        y = K.batch_dot(z, mV)

        return y

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2, \
            'expected to receive 2 tensors as input. Multiple inputs are not specified'
        assert len(input_shape) == 2, \
            f'expected to receive 2 tensors as input. Received {len(input_shape)}'
        shape_h, shape_m = input_shape

        return None, shape_h[1], self.weight_layer.w_k.shape[1]
