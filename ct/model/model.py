from keras import activations
from keras import regularizers
from keras import callbacks
from keras import backend as K
from keras.models import Model
from keras.models import Sequential as SequentialModel
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, Input


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
                 layers=1,
                 sequence_length=512,
                 memory_size=512,
                 compressed_memory_size=512,
                 name='CompressiveTransformer',
                 **kwargs):
        if layers > 1:
            raise NotImplementedError()

        self.sequence_length = sequence_length
        self.memory = K.zeros(shape=memory_size)
        self.compressed_memory = K.zeros(shape=compressed_memory_size)
        self.reconstruction_model = AttentionReconstruction()

        pass

        super().__init__(*args, inputs=[input_layer], outputs=[output_layer], name=name, **kwargs)

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        super().train_on_batch(x=x,
                               y=y,
                               sample_weight=sample_weight,
                               class_weight=class_weight,
                               reset_metrics=reset_metrics)
        old_mem, new_cm = self.update_memory()

        h = K.function([self.input], self.head)(x)

        self.reconstruction_model.train_on_batch(x=[h, old_mem],
                                                 y=new_cm,
                                                 sample_weight=sample_weight,
                                                 reset_metrics=reset_metrics)

    def update_memory(self):
        # Oldest memories to be forgotten
        # Compress Oldest memories by factor c
        # Update memory
        # Update compressed memory
        
        pass


class AttentionReconstruction(Model):
    _max_pool = ['max-pool', 'max_pool', 'max pool', 'max']
    _1d_conv = ['1d-conv', '1d_conv', '1d conv', 'conv']
    _all_methods = _max_pool[:1] + _1d_conv[:1]

    def __init__(self,
                 input_shapes,
                 heads,
                 *args,
                 compression='1d-conv',
                 compression_rate=3,
                 name='AttentionReconstruction',
                 **kwargs):
        assert isinstance(heads, list)
        # h
        # old_mem
        # heads
        h_shape, old_mem_shape = input_shapes

        h = Input(h_shape[1:])
        old_mem = Input(old_mem_shape[1:])

        if compression in self._max_pool:
            raise NotImplementedError()
        elif compression in self._1d_conv:
            from keras.layers import Conv1D

            output_layer = Conv1D(filters=128,
                                  kernel_size=compression_rate,
                                  stride=compression_rate,
                                  activation='relu')(old_mem)
        else:
            raise NotImplementedError(f'unsupported compression: {compression}. Select one from {self._all_methods}')

        super().__init__(*args, inputs=[h, old_mem], outputs=[output_layer], name=name, **kwargs)
        self.heads = heads
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
        if len(self.heads) > 1:
            raise NotImplementedError()

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
