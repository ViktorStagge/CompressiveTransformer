from keras import activations
from keras import regularizers
from keras import callbacks
from keras import backend as K
from keras.models import Model
from keras.models import Sequential as SequentialModel
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, Input, Conv1D, Add, concatenate as Concatenate


from model.layers import MultiHeadAttention, ScaledDotProductAttention, LayerNormalization
from model.layers.attention import ContentBasedAttention_CT, content_based_attention


def model():
    model = naive_model()
    return model


def naive_model():
    model = SequentialModel()
    model.add(Embedding(input_dim=1000, output_dim=100))
    model.add(LSTM(units=64))
    model.add(Dense(100))
    model.add(Dropout(rate=0.2))
    model.add(Dense(50))

    model.compile(optimizer='Adam',
                  loss='mse')
    return model


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
    _2_hL_0 = LayerNormalization(units=dense_units)(_2_h_0)

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
        _2_hL_0 = LayerNormalization(units=dense_units)(_2_h_0)

        # Dense Output
        output_layer = Dense(output_size, activation='softmax', name='output_layer')(_2_hL_0)

        super().__init__(inputs=[input_layer], outputs=[output_layer], *args, **kwargs)


class CompressiveTransformer(Model):
    def __init__(self,
                 *args,
                 sequence_length=512,
                 memory_size=512,
                 compressed_memory_size=512,
                 batch_size=1,
                 d_layers=1,
                 d_heads=2,
                 d_model=1024,  #
                 d_k=None,
                 d_mlp_hidden=None,  # 3072
                 vocab_size=20000,
                 name='CompressiveTransformer',
                 **kwargs):
        if d_layers > 1:
            raise NotImplementedError()
        if batch_size > 1:
            raise NotImplementedError()
        if d_k is None:
            d_k = d_model  # // d_heads
        if d_mlp_hidden is None:
            d_mlp_hidden = d_model
        self.memory = K.zeros(shape=(batch_size, memory_size, d_model),
                              name='memory')
        self.compressed_memory = K.zeros(shape=(batch_size, compressed_memory_size, d_model),
                                         name='compressed_memory')

        # Build the internal model structure
        x = Input(shape=(sequence_length,), name='x')
        memory = Input(shape=self.memory.shape[1:], name='memory')
        compressed_memory = Input(shape=self.compressed_memory.shape[1:], name='compressed_memory')

        h = Embedding(input_dim=vocab_size,
                      output_dim=d_model,
                      embeddings_initializer='uniform')(x)

        # TODO: h = h_token + h_pos
        concat_memory = Concatenate([memory, compressed_memory], axis=1)
        print(concat_memory)

        # #### Multi Head Attention #####
        sdpa_layers = [ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_model) for _ in range(d_heads)]
        sdpa = [layer([h, concat_memory]) for layer in sdpa_layers]

        mha_layer = MultiHeadAttention(d_heads=d_heads,
                                       d_model=d_model,
                                       d_k=d_k,
                                       d_v=d_model,
                                       name='multihead_attention_L0')
        mha = mha_layer(sdpa)
        mha_skip = Add(name='mha_skip_L0')([h, mha])
        # #### #################### #####

        a = LayerNormalization(units=sequence_length*d_model, name='mha_layer_norm_L0')(mha_skip)

        mlp = Dense(units=d_mlp_hidden, name='mlp_hidden_0_L0')(a)
        mlp = Dense(units=d_model, activation='softmax', name='mlp_L0')(mlp)
        mlp_skip = Add(name='mlp_skip_L0')([mlp, a])

        h_next = LayerNormalization(units=sequence_length*d_model, name='mlp_layer_norm_L0')(mlp_skip)

        output_layer = h_next

        # super().__init__(*args, inputs=[x], outputs=[output_layer], name=name, **kwargs)
        super().__init__(*args,
                         inputs=[x, memory, compressed_memory],
                         outputs=[output_layer],
                         name=name,
                         **kwargs)
        # layer outputs
        self._h = [h]
        self._sdpa = sdpa
        self._mha = mha
        self._mha_ship = mha_skip
        self._a = a
        self._mlp = mlp
        self._mlp_skip = mlp_skip
        # layers
        self._sdpa_layers = sdpa_layers
        self._mha_layer = mha_layer
        # settings
        self.sequence_length = sequence_length
        self.memory_size = memory_size
        self.compressed_memory_size = compressed_memory_size
        self.vocab_size = vocab_size
        self.d_layers = d_layers
        self.d_model = d_model
        self.d_heads = d_heads
        self.d_k = d_k
        self.d_mlp_hidden = d_mlp_hidden

        self.sequence_length = sequence_length
        self.reconstruction_model = AttentionReconstruction(input_shape=[self.memory.shape,
                                                                         self.compressed_memory.shape],
                                                            heads=self._sdpa_layers[:1])

    def comile(self,
               optimizer,
               loss=None,
               metrics=None,
               loss_weights=None,
               reconstruction_optimizer='Adam',
               reconstruction_metrics=None,
               **kwargs):
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        **kwargs)
        self.reconstruction_model.compile(optimizer=reconstruction_optimizer,
                                          metrics=reconstruction_metrics)

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        x_input = [x, self.memory, self.compressed_memory]
        super().train_on_batch(x=x_input,
                               y=y,
                               sample_weight=sample_weight,
                               class_weight=class_weight,
                               reset_metrics=reset_metrics)

        h = K.function([self.input], self._h[0])(x_input)
        old_mem, new_cm = self.update_memory(h=h)

        self.reconstruction_model.train_on_batch(x=[h, old_mem],
                                                 y=new_cm,
                                                 sample_weight=sample_weight,
                                                 reset_metrics=reset_metrics)

    def update_memory(self, h):
        old_mem = self.memory[:, self.sequence_length, :]
        new_cm = self.reconstruction_model(inputs=[h, old_mem])

        self.memory = K.concatenate([self.memory[:, self.sequence_length:, :], h], axis=0)
        self.compressed_memory = K.concatenate([self.compressed_memory[:, self.sequence_length:, :], new_cm], axis=0)
        return old_mem, new_cm


class AttentionReconstruction(Model):
    _max_pool = ['max-pool', 'max_pool', 'max pool', 'max']
    _1d_conv = ['1d-conv', '1d_conv', '1d conv', 'conv']
    _all_methods = _max_pool[:1] + _1d_conv[:1]

    def __init__(self,
                 input_shape,
                 heads,
                 *args,
                 compression='1d-conv',
                 compression_rate=3,
                 name='AttentionReconstruction',
                 **kwargs):
        assert isinstance(heads, list)
        if len(heads) > 1:
            raise NotImplementedError()
        # h
        # old_mem
        # heads
        h_shape, old_mem_shape = input_shape

        h = Input(batch_shape=h_shape)
        old_mem = Input(batch_shape=old_mem_shape)

        print(h_shape)
        print(old_mem_shape)

        if compression in self._max_pool:
            raise NotImplementedError()
        elif compression in self._1d_conv:
            filters = kwargs.get('conv_filters', 128)

            output_layer = Conv1D(filters=filters,
                                  kernel_size=compression_rate,
                                  strides=compression_rate,
                                  activation='relu')(old_mem)
        else:
            raise NotImplementedError(f'unsupported compression: {compression}. '
                                      f'Select one from {self._all_methods}')

        super().__init__(*args, inputs=[h, old_mem], outputs=[output_layer], name=name, **kwargs)
        self.heads = heads
        self.compression = compression
        self.compression_rate = compression_rate
        self._current_batch_h = None
        self._current_batch_old_mem = None
        self._current_batch_new_cm = None

    def compile(self,
                optimizer,
                loss='attention_reconstruction',
                metrics=None,
                loss_weights=None,
                **kwargs):
        assert loss == 'attention_reconstruction'
        super().compile(optimizer=optimizer,
                        loss=self.attention_reconstruction_loss(),
                        metrics=metrics,
                        loss_weights=loss_weights)

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        self._current_batch_h = [x[0]]
        self._current_batch_old_mem = [x[1]]
        self._current_batch_new_cm = [y]
        super().train_on_batch(x=x,
                               y=y,
                               sample_weight=sample_weight,
                               class_weight=class_weight,
                               reset_metrics=reset_metrics)

    def attention_reconstruction_loss(self):

        def _attention_reconstruction_loss(y_train, y_true):
            assert len(self.heads) == 1
            assert len(self._current_batch_old_mem) == 1
            assert len(self._current_batch_new_cm) == 1
            loss = 0

            for head, h, old_mem, new_cm in zip(self.heads, self._current_batch_h,
                                                self._current_batch_old_mem, self._current_batch_new_cm):
                old_attention = content_based_attention(h=h, m=old_mem, w_q=head.w_q, w_k=head.w_k, w_v=head.w_v)
                new_attention = content_based_attention(h=h, m=new_cm, w_q=head.w_q, w_k=head.w_k, w_v=head.w_v)
                loss_head = (old_attention - new_attention)

                loss += loss_head
            return loss

        return _attention_reconstruction_loss
