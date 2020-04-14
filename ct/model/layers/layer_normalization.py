from keras import layers
from keras import backend as K
from keras.layers import Layer


class LayerNormalization(Layer):

    def __init__(self, units, bias=1, gain=1, **kwargs):
        self.units = units
        self.bias = bias
        self.gain = gain
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, tuple) and len(input_shape) == 2, \
            f'received input_shape={input_shape}. Only 1-dim input_shape is supported.'
        assert input_shape == (None, self.units), \
            f'received input_shape={input_shape}. Expected input_shape=(None, {self.units})'

        super().build(input_shape)

    def call(self, x, **kwargs):
        mean = K.sum(x) / self.units
        std_dev = K.sqrt(K.sum(K.square(x - mean)) / self.units)

        y = (x - mean) / std_dev
        if self.gain:
            y *= self.gain
        if self.bias:
            y += self.bias
        return y

    def compute_output_shape(self, input_shape):
        return input_shape
