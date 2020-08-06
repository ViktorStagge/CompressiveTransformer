import warnings
import numpy as np

from keras import activations
from keras import regularizers
from keras import callbacks
from keras import backend as K
from keras.models import Model
from keras.models import Sequential as SequentialModel
from keras.layers import Embedding, \
                         LSTM, \
                         Dense, \
                         Dropout, \
                         Flatten, \
                         Input, \
                         Conv1D, \
                         Add, \
                         Reshape, \
                         Lambda, \
                         concatenate as Concatenate
from omegaconf import OmegaConf
from typing import List


from model.layers import MultiHeadAttention, \
                         ScaledDotProductAttention, \
                         LayerNormalization, \
                         ReverseEmbedding, \
                         RelativeEncoding
from model.layers.attention import ContentBasedAttention_CT, \
                                   content_based_attention
from model.optimizers import get_optimizer
from config.default import default_config as config


def naive_multiehead_model(d_heads=2,
                           d_model=128,
                           d_k=16,
                           d_v=128,
                           input_vocab_size=15000,
                           input_sequence_length=None,
                           embedding_size=None,
                           dense_units=10,
                           output_size=3):
    if embedding_size is None:
        embedding_size = d_model
    if input_sequence_length is None:
        input_sequence_length = d_model

    # Embedding
    input_layer = Input(shape=(input_sequence_length,))
    _0_x = Embedding(input_dim=input_vocab_size, output_dim=embedding_size)(input_layer)

    # Multi Head Attention
    _1_s = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_model)(_0_x) for _ in range(d_heads)]
    _1_m = MultiHeadAttention(d_heads=d_heads, d_model=d_model, d_k=d_k, d_v=d_v)(_1_s)

    # Dense
    _2_f = Flatten()(_1_m)
    _2_h_0 = Dense(units=dense_units, name='hidden_0')(_2_f)
    _2_hL_0 = LayerNormalization()(_2_h_0)

    # Dense Output
    output_layer = Dense(output_size, activation='softmax', name='output_layer')(_2_hL_0)

    #
    # Compile Model
    model = Model(inputs=[input_layer],
                  outputs=[output_layer])
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class MultiHeadAttentionModel(Model):
    def __init__(self,
                 *args,
                 d_heads=2,
                 d_model=128,
                 d_k=16,
                 d_v=128,
                 input_vocab_size=15000,
                 input_sequence_length=None,
                 embedding_size=None,
                 dense_units=10,
                 output_size=3,
                 **kwargs):
        if embedding_size is None:
            embedding_size = d_model
        if input_sequence_length is None:
            input_sequence_length = d_model
        if 'name' not in kwargs:
            kwargs['name'] = type(MultiHeadAttentionModel).__name__

        # Embedding
        input_layer = Input(shape=(input_sequence_length,))
        _0_x = Embedding(input_dim=input_vocab_size, output_dim=embedding_size)(input_layer)

        # Multi Head Attention
        _1_s = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_model)(_0_x) for _ in range(d_heads)]
        _1_m = MultiHeadAttention(d_heads=d_heads, d_model=d_model, d_k=d_k, d_v=d_v)(_1_s)

        # Dense
        _2_f = Flatten()(_1_m)
        _2_h_0 = Dense(units=dense_units, name='hidden_0')(_2_f)
        _2_hL_0 = LayerNormalization()(_2_h_0)

        # Dense Output
        output_layer = Dense(output_size, activation='softmax', name='output_layer')(_2_hL_0)

        super().__init__(inputs=[input_layer], outputs=[output_layer], *args, **kwargs)


class CompressiveTransformer(Model):
    def __init__(self,
                 *args,
                 sequence_length,
                 memory_size=512,
                 compressed_memory_size=512,
                 compression_rate=3,
                 batch_size=1,
                 d_layers=1,
                 d_heads=2,
                 d_model=1024,
                 d_k=None,
                 d_mlp_hidden=None,  # 3072
                 vocab_size=20000,
                 output_size=None,
                 dropout_probability=0.1,
                 use_relative_encoding=None,
                 name='CompressiveTransformer',
                 **kwargs):
        assert memory_size >= sequence_length, \
            'Memory has to be longer than the sequence length'
        assert compressed_memory_size >= sequence_length // compression_rate, \
            'Compressed memory has to be longer than the compressed sequence length'
        if d_layers <= 0:
            warnings.warn('d_layers is 0, not using any layers of the Compressive Transformer.')
        if d_k is None:
            d_k = d_model  # // d_heads
        if d_mlp_hidden is None:
            d_mlp_hidden = d_model
        if use_relative_encoding is None:
            use_relative_encoding = config.feature_relative_encoding
        memory = np.zeros(shape=(batch_size, d_layers, memory_size, d_model))
        compressed_memory = np.zeros(shape=(batch_size, d_layers, compressed_memory_size, d_model))

        # Build the internal model structure
        x = Input(shape=(sequence_length,),
                  name='x')
        x_memory = Input(shape=memory.shape[1:],
                         name='memory')
        x_compressed_memory = Input(shape=compressed_memory.shape[1:],
                                    name='compressed_memory')

        embedding_layer = Embedding(input_dim=vocab_size,
                                    output_dim=d_model,
                                    embeddings_initializer='uniform',
                                    name='word_embedding')
        e_w = embedding_layer(x)

        if use_relative_encoding:
            e_r = RelativeEncoding(batch_size=batch_size,
                                   verbose=True,
                                   name='relative_encoding')(e_w)

            e = Add(name='h_L0')([e_w, e_r])
        else:
            e = e_w
        h = Dropout(rate=dropout_probability, name='dropout_embedding')(e)

        _hs = []
        _sdpa_layers = []
        for i in range(d_layers):
            _hs.append(h)

            _mem = Lambda(lambda mem: mem[:, i, :, :], name=f'select_memory_L{i}')(x_memory)
            _comp_mem = Lambda(lambda mem: mem[:, i, :, :], name=f'select_compressed_memory_L{i}')(x_compressed_memory)
            h_tilde = Concatenate([_comp_mem, _mem, h], axis=1, name=f'h_tilde_L{i}')

            # #### Multi Head Attention #####
            sdpa_layers = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_model) for _ in range(d_heads)]
            _sdpa_layers.append(sdpa_layers)
            sdpa = [sdpa_layer([h, h_tilde]) for sdpa_layer in sdpa_layers]

            mha = MultiHeadAttention(d_heads=d_heads,
                                     d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_model,
                                     name=f'multihead_attention_L{i}')(sdpa)
            mha_skip = Add(name=f'mha_skip_L{i}')([h, mha])
            a = LayerNormalization(name=f'mha_layer_norm_L{i}')(mha_skip)
            # # #### #################### #####

            mlp_hidden = Dense(units=d_mlp_hidden,
                               activation='relu',
                               name=f'mlp_hidden_0_L{i}')(a)
            mlp = Dense(units=d_model,
                        activation=None,
                        name=f'mlp_no_activation_L{i}')(mlp_hidden)
            mlp_drop = Dropout(rate=dropout_probability, name=f'dropout_L{i}')(mlp)
            mlp_skip = Add(name=f'mlp_skip_L{i}')([mlp_drop, a])
            
            h = LayerNormalization(name=f'h_L{i+1}')(mlp_skip)  # h, for L_{i+1}

        encoder_output = h  # intermediate output

        reverse_embedding_layer = ReverseEmbedding(embedding_layer,
                                                   activation='softmax',
                                                   name='output')
        _z = reverse_embedding_layer(encoder_output)
        outputs = _z

        super().__init__(*args,
                         inputs=[x, x_memory, x_compressed_memory],
                         outputs=outputs,
                         name=name,
                         **kwargs)

        # Attention Reconstruction Model (Model for compressing memory)
        # Memory
        self.memory = memory
        self.compressed_memory = compressed_memory
        
        # layer outputs
        self._h = _hs
        
        # layers
        self._sdpa_layers = _sdpa_layers
        
        # settings
        self.sequence_length = sequence_length
        self.memory_size = memory_size
        self.compressed_memory_size = compressed_memory_size
        self.compression_rate = compression_rate
        self.compressed_sequence_length = sequence_length // compression_rate
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.d_layers = d_layers
        self.d_model = d_model
        self.d_heads = d_heads
        self.d_k = d_k
        self.d_mlp_hidden = d_mlp_hidden

    def _create_reconstruction_models_(self):
        self.reconstruction_models = [AttentionReconstruction(input_shape=[self._h[i].shape,
                                                                           self._h[i].shape],
                                                              heads=self._sdpa_layers[i][:1],
                                                              compression_rate=self.compression_rate)
                                      for i in range(self.d_layers)]

    def compile(self,
                optimizer,
                loss=None,
                metrics=None,
                loss_weights=None,
                reconstruction_optimizer='Adam',
                reconstruction_metrics=None,
                **kwargs):
        super().compile(optimizer=get_optimizer(optimizer),
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        **kwargs)
        self._create_reconstruction_models_()
        for reconstruction_model in self.reconstruction_models:
            reconstruction_model.compile(optimizer=reconstruction_optimizer,
                                         metrics=reconstruction_metrics)

    def train_on_batch(self,
                       x,
                       y,
                       sample_weight=None,
                       class_weight=None,
                       reset_metrics=True):
        """
        Arguments:
            x:
            y:
            sample_weight:
            class_weight:
            reset_metrics:

        Returns:
            loss:
            loss_ar:
        """
        loss = super().train_on_batch(x=x,
                                      y=y,
                                      sample_weight=sample_weight,
                                      class_weight=class_weight,
                                      reset_metrics=reset_metrics)
        
        h = K.function(self.input, self._h)(x)
        
        old_mem, new_cm = self.update_memory(h=h)
        
        loss_ar = 0
        for reconstruction_model, _h, _om, _ncm in zip(self.reconstruction_models, h, old_mem, new_cm):
            loss_ar += reconstruction_model.train_on_batch(x=[_h, _om],
                                                           y=_ncm,
                                                           sample_weight=sample_weight,
                                                           reset_metrics=reset_metrics)
        loss_ar = loss_ar / max(1, self.d_layers)
        return loss  # , loss_ar

    def summary(self, line_length=None, positions=None, print_fn=None):
        super().summary(line_length=line_length,
                        positions=positions,
                        print_fn=print_fn)
        
        if hasattr(self, 'reconstruction_model') \
                and self.reconstruction_models is not None:
            if print_fn is None:
                print('\n\n\n')
            self.reconstruction_models[0].summary(line_length=line_length,
                                                  positions=positions,
                                                  print_fn=print_fn)

    def update_memory(self,
                      h: List[np.ndarray]):
        # breaks on d_layers > 1
        # breaks on dims Input > 3 ...
        old_mem = self.memory[:, :, :self.sequence_length, :]
        old_mem = [old_mem[:, i, :, :] for i in range(self.d_layers)]
        
        # new_cm = [reconstruction_model(inputs=[K.variable(_h), K.variable(_om)])
        #           for reconstruction_model, _h, _om in zip(self.reconstruction_models, h, old_mem)]
        # new_cm = [K.eval(_ncm) for _ncm in new_cm]
        new_cm = [self.compressed_memory[:, i, :self.compressed_sequence_length, :] for i in range(self.d_layers)]
        
        for i, (_h, _ncm) in enumerate(zip(h, new_cm)):
            self.memory[:, i, :-self.sequence_length, :] = self.memory[:, i, self.sequence_length:, :]
            self.memory[:, i, -self.sequence_length:, :] = _h
            
            self.compressed_memory[:, i, :-self.compressed_sequence_length, :] = self.compressed_memory[:, i, self.compressed_sequence_length:, :]
            self.compressed_memory[:, i, -self.compressed_sequence_length:, :] = _ncm

        return old_mem, new_cm


_max_pool = ['max-pool', 'max_pool', 'max pool', 'max']
_1d_conv = ['1d-conv', '1d_conv', '1d conv', 'conv']
_all_compressions = _max_pool[:1] + _1d_conv[:1]


class AttentionReconstruction(Model):

    def __init__(self,
                 input_shape,
                 heads,
                 *args,
                 compression='1d-conv',
                 compression_rate=3,
                 name='AttentionReconstruction',
                 verbose=False,
                 **kwargs):
        assert isinstance(heads, list)
        if len(heads) > 1:
            raise NotImplementedError()
        # heads

        h_shape, old_mem_shape = input_shape
        assert h_shape == old_mem_shape

        h = Input(batch_shape=h_shape, name='ar_h')
        old_mem = Input(batch_shape=old_mem_shape, name='ar_old_mem')

        if compression in _max_pool:
            raise NotImplementedError()
        elif compression in _1d_conv:
            filters = kwargs.get('conv_filters', 128)
            activation = kwargs.get('conv_activation', 'relu')

            output_layer = Conv1D(filters=filters,
                                  kernel_size=compression_rate,
                                  strides=compression_rate,
                                  activation=activation,
                                  name='ar_conv1D')
            output = output_layer(old_mem)
        else:
            raise ValueError(f'unsupported compression: {compression}. '
                             f'Select one from {_all_compressions}')

        super().__init__(*args, inputs=[h, old_mem], outputs=output, name=name, **kwargs)
        self.heads = heads
        self.compression = compression
        self.compression_rate = compression_rate
        self._current_batch = dict(h=[h],
                                   old_mem=[old_mem],
                                   new_cm=[output])
        self.verbose = verbose
        self._custom_layers = dict(output=output_layer)

        if verbose:
            print(self.summary())

    def compile(self,
                optimizer,
                loss='attention_reconstruction',
                metrics=None,
                loss_weights=None,
                **kwargs):
        if loss == 'attention_reconstruction':
            loss = self.attention_reconstruction_loss()
        else:
            warnings.warn('using non-standard loss for AttentionReconstruction', RuntimeWarning)

        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        **kwargs)

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        self._current_batch['h'] = [x[0]]
        self._current_batch['old_mem'] = [x[1]]
        self._current_batch['new_cm'] = [y]

        loss = super().train_on_batch(x=x,
                                      y=y,
                                      sample_weight=sample_weight,
                                      class_weight=class_weight,
                                      reset_metrics=reset_metrics)
        return loss

    def attention_reconstruction_loss(self):

        def _attention_reconstruction_loss(y_true, y_pred):
            # assert len(self.heads) == 1
            # assert len(self._current_batch_old_mem) == 1
            # assert len(self._current_batch_new_cm) == 1
            # print('   calculating loss...')
            return (y_true - y_pred) ** 2

        #             for head, h, old_mem, new_cm in zip(self.heads,
        #                                                 self._current_batch['h'],
        #                                                 self._current_batch['old_mem'],
        #                                                 self._current_batch['new_cm']):
        #                 print(h, old_mem, head.w_q, head.w_k, head.w_v, sep='\n')
        #                 old_attention = content_based_attention(h=h, m=old_mem, w_q=head.w_q, w_k=head.w_k, w_v=head.w_v)
        #                 new_attention = content_based_attention(h=h, m=new_cm, w_q=head.w_q, w_k=head.w_k, w_v=head.w_v)
        #                 loss_head = (old_attention - new_attention)

        #                 loss += loss_head

        #             print((y_true - y_pred).shape)
        #             print(self._current_batch['h'][0])
        #             # # works
        #             # return y_true - y_pred
        #             return y_true - self._current_batch['new_cm'][0]

        #             # # doesn't work
        #             # return K.zeros(shape=y_pred.shape)

        return _attention_reconstruction_loss

