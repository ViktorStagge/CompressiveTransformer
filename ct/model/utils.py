from keras import backend as K


def euclidean_norm(x):
    """Euclidean norm for a Keras Tensor.

    Arguments:
        x: Keras tensor

    Returns:
        euclidean_norm: euclidean norm for the inputted tensor
    """
    return K.sqrt(K.sum(K.square(x)))


def cosine_similarity(x1, x2):
    """Cosine similarity between two Keras Tensors.
    Uses Euclidean norm.

    Arguments:
        x1: Keras Tensor
        x2: Keras Tensor

    Returns:
        cosine_similarity: cosine similairty between input tensors

    """
    x1_norm = euclidean_norm(x1)
    x2_norm = euclidean_norm(x2)
    cosine_similarity = K.dot(x1, x2) / x1_norm / x2_norm
    return cosine_similarity
