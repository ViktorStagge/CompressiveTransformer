import warnings

from keras import backend as K


def _content_based_attention(h, m, w_q, w_k, w_v):
    warnings.warn('relic from previous iteration, no longer in use.', DeprecationWarning)
    hQ = K.dot(h, w_q)
    mK = K.dot(m, w_k)
    mV = K.dot(m, w_v)

    z = K.batch_dot(hQ, mK)
    z = K.softmax(z)
    y = K.batch_dot(z, mV)

    return y