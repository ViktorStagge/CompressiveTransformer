import numpy as np

from keras import layers
from keras import backend as K
from keras.layers import Layer


class LayerNormalization(Layer):

    def __init__(self,
                 use_bias=True,
                 use_gain=True,
                 verbose=False,
                 **kwargs):
        self.use_gain = use_gain
        self.use_bias = use_bias
        self.units = None
        self.gain = None
        self.bias = None
        self.verbose = verbose

        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, tuple), \
            f'received input_shape={input_shape}. Expected a tuple (i.e. single input).'
        # assert np.prod(input_shape[1:]) == self.units, \
        #     f'received input_shape={input_shape}. Expected total of units={self.units}'
        self.units = np.prod(input_shape[1:])

        self.gain = self.add_weight(name='gain',
                                    shape=input_shape[1:],
                                    initializer='uniform',
                                    trainable=self.use_gain)
        self.bias = self.add_weight(name='bias',
                                    shape=input_shape[1:],
                                    initializer='uniform',
                                    trainable=self.use_bias)
        super().build(input_shape)

    def call(self, x, **kwargs):
        mean = K.sum(x) / self.units
        std_dev = K.sqrt(K.sum(K.square(x - mean)) / self.units)

        y = (x - mean) / std_dev
        if self.use_gain:
            y *= self.gain
        if self.use_bias:
            y += self.bias
        if self.verbose:
            print(f'LayerNormalization:\n'
                  f'    x.shape={x.shape}'
                  f'    y.shape={y.shape}')
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(use_gain=self.use_gain,
                      use_bias=self.use_bias,
                      verbose=self.verbose)
        return config
