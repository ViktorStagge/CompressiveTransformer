from keras import backend as K


def attention_reconstruction_loss(model, input_after_embedding):
    loss = 0


    h = input_after_embedding
    old_memory = model.memory.copy()
    q, k, v = model.

    pass


def _content_based_attention(h, m, w_q, w_k, w_v):
    hQ = K.dot(h, w_q)
    mK = K.dot(m, w_k)
    mV = K.dot(m, w_v)

    z = K.batch_dot(hQ, mK)
    z = K.softmax(z)
    y = K.batch_dot(z, mV)

    return y