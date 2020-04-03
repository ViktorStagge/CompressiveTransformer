from keras import backend as K
from keras.layers import Layer
from keras.layers import MaxPool1D
from keras.layers import MaxPool2D
from keras.layers import AvgPool1D
from keras.layers import AvgPool2D
from keras.layers import Conv1D


def compression_layer(compression, **kwargs):
    max_pool = ['max', 'max_pool', 'max-pool']
    mean_pool = ['mean', 'mean_pool', 'mean-pool', 'avg', 'avg_pool', 'avg-pool']
    convolution = ['conv', 'convolution', 'conv1d']
    dilated_convolution = ['dilated', 'dilated-conv', 'dilated-convolution', 'dilated-convolutions']
    most_used = ['most', 'most_used', 'most-used']
    all_compressions = ['max_pool', 'mean_pool', 'convolution', 'dilated_convolution', 'most-used']
    if isinstance(compression, str):
        compression = compression.lower()

    if compression in max_pool:
        layer = MaxPool1D(**kwargs)
    elif compression in mean_pool:
        layer = AvgPool1D(**kwargs)
    elif compression in convolution:
        assert 'filters' in kwargs or 'units' in kwargs, \
            'convolution-compression requries key-word argument `filters`'
        assert 'kernel_size' in kwargs, \
            'convolution-compression requries key-word argument `kernel_size`'
        filters = kwargs.get('filters') or kwargs.get('units')

        layer = Conv1D(filters=filters,
                       kernel_size=3,
                       **kwargs)
    elif compression in dilated_convolution:
        raise NotImplementedError('`dilated-convolution compression` is not implemented.')
    elif compression in most_used:
        raise NotImplementedError('`most-used compression` is not implemented.')
    else:
        raise ValueError(f'unexpected compression: {compression}. '
                         f'Select from [{all_compressions}]')
    return layer
