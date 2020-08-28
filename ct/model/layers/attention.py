import warnings
import numpy as np

from keras import layers
from keras import backend as K
from keras.layers import Layer
from omegaconf import OmegaConf

from model.utils import cosine_similarity


_default = OmegaConf.create(dict(d_k=64,
                                 d_v=64,
                                 d_model=512,
                                 d_heads=2))


class ScaledDotProductAttention(Layer):
    def __init__(self,
                 d_k=None,
                 d_q=None,
                 d_v=None,
                 d_model=None,
                 verbose=False,
                 **kwargs):
        if verbose:
            print(f'#### INIT [{self.__class__.__name__}] ####')
        if d_q not in (d_k, None):
            warnings.warn('using different dimensions for `keys` [d_k] and `queries` [d_q]. '
                          'Functionality is not tested.')
        super().__init__(**kwargs)

        self.verbose = verbose
        self.d_model = d_model or _default.d_model
        self.d_k = d_k or _default.d_k
        self.d_q = d_q or self.d_k
        self.d_v = d_v or _default.d_v
        self.w_q = None
        self.w_k = None
        self.w_v = None

    def build(self, input_shape):
        if self.verbose:
            print(f'#### BUILD [{self.__class__.__name__}] ####')
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

        # assert [...],
        #     f'unexpected input shapes received for: {input_shape}, expected:\n' \
        #     f'd_model= {self.d_model}\n' \
        #     f'd_k=d_q= {self.d_k}\n' \
        #     f'd_v    = {self.d_v}'

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
        if self.verbose:
            print(f'#### CALL [{self.__class__.__name__}] ####')
        if isinstance(x, list):
            if len(x) == 3:
                q, k, v = x
            elif len(x) == 2:
                q, k = x
                v = k
            else:
                raise NotImplementedError('ScaledDotProductAttention is not implemented for 3D-input or above')
        else:
            q = x
            k = x
            v = x

        # assert [...], \
        #     f'unexpected input shape received for `Q: Query`: {q.shape}'
        # assert [...]

        q = K.dot(q, self.w_q)  # (batch, n_s, d_model) x (d_model, d_q)
        v = K.dot(v, self.w_v)
        k = K.dot(k, self.w_k)
        if len(k.shape) == 3:
            k_transpose = K.permute_dimensions(k, pattern=(0, 2, 1))
        else:
            raise NotImplementedError('expected 3 dimensions')

        z = K.batch_dot(q, k_transpose)
        z = z / np.sqrt(self.d_k)  # optional Mask for decoder after this line
        z = K.softmax(z)
        y = K.batch_dot(z, v)

        if self.verbose:
            print(f'Scaled Dot Product Attention:\n'
                  f'    q.shape={q.shape},\n'
                  f'    k.shape={k.shape},\n'
                  f'    v.shape={v.shape}\n'
                  f'    k_transpose.shape={k_transpose.shape}\n'
                  f'    y.shape={y.shape}')

        return y

    def compute_output_shape(self, input_shape):
        if self.verbose:
            print('#### COMPUTE OUTPUT SHAPE ####')
        if isinstance(input_shape, list):
            assert len(input_shape) in (2, 3), \
                f'received input_shape of length `{len(input_shape)}`, expected length in (2, 3)'
            if len(input_shape) == 3:
                shape_q, shape_k, shape_v = input_shape
            if len(input_shape) == 2:
                shape_q, shape_k = input_shape
                shape_v = shape_k
        else:
            shape_q = input_shape
            shape_k = input_shape
            shape_v = input_shape

        output_shape = (None, shape_q[1], self.d_v)
        if self.verbose:
            print(f'   compute_output_shape: {output_shape}')
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update(d_model=self.d_model,
                      d_k=self.d_k,
                      d_q=self.d_q,
                      d_v=self.d_v,
                      verbose=self.verbose)
        return config


class MultiHeadAttention(Layer):
    def __init__(self,
                 d_heads=None,
                 d_k=None,
                 d_q=None,
                 d_v=None,
                 d_model=None,
                 sequence_length=None,
                 verbose=False,
                 **kwargs):
        if d_q not in (d_k, None):
            warnings.warn('using different dimensions for `keys` [d_k] and `queries` [d_q]. '
                          'Functionality is not tested.')
        super().__init__(**kwargs)

        self.verbose = verbose
        self.d_heads = d_heads or _default.d_heads
        self.d_model = d_model or _default.d_model
        self.d_k = d_k or _default.d_k
        self.d_q = self.d_k
        self.d_v = d_v or _default.d_v
        self.sequence_length = sequence_length or self.d_model
        self.w_o = None

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

        if self.verbose:
            print(f'{self.__class__.__name__}:\n'
                  f'    inputs={len(x)}\n'
                  f'    w_o.shape={self.w_o.shape}\n'
                  f'    x_i.shape={x[0].shape}\n'
                  f'    z.shape={z.shape}\n'
                  f'    y.shape={y.shape}')

        return y

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        head_shape = input_shape[-1]
        assert len(head_shape) == 3
        assert head_shape[1] == self.sequence_length
        assert head_shape[2] == self.d_v

        return None, self.sequence_length, self.d_model

    def get_config(self):
        config = super().get_config()
        config.update(d_heads=self.d_heads,
                      d_model=self.d_model,
                      sequence_length=self.sequence_length,
                      d_k=self.d_k,
                      d_q=self.d_q,
                      d_v=self.d_v,
                      verbose=self.verbose)
        return config


class ContentBasedAttention(Layer):
    """Content Based Attention as defined in "Neural Turing Machines" by Graves et. al.
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


def content_based_attention(h, m, w_q, w_k, w_v, normalize=False, d_k=None):
    hq = K.dot(h, w_q)
    mk = K.dot(m, w_k)
    mv = K.dot(m, w_v)

    if len(mk.shape) == 3:
        mk_transpose = K.permute_dimensions(mk, pattern=(0, 2, 1))
    else:
        raise NotImplementedError('expected 3 dimensions')

    z = K.batch_dot(hq, mk_transpose)
    if normalize:
        z = z / np.sqrt(d_k)
    z = K.softmax(z)
    y = K.batch_dot(z, mv)
    return y


def content_based_attention_numpy(h: np.ndarray,
                                  m: np.ndarray,
                                  w_q: np.ndarray,
                                  w_k: np.ndarray,
                                  w_v: np.ndarray):
    from scipy.special import softmax

    hQ = np.dot(h, w_q)
    mK = np.dot(m, w_k)
    mV = np.dot(m, w_v)

    z = np.tensordot(hQ, mK, axes=(0, 1))
    z = softmax(z, axis=-1)
    y = np.tensordot(z, mV, axes=(0, 1))

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
