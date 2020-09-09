from keras import optimizers
from omegaconf import OmegaConf
from dataclasses import dataclass


@dataclass
class Adam:
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    clipnorm: float = 0.1


def get_optimizer(method, **kwargs):
    """Retrieves the specified optimizer.
    Defaults to `Adam` with the parameters as specified by Rae et. al.
    """
    if isinstance(method, str):
        method = method.lower()

    if method in ['adam']:
        config = OmegaConf.structured(Adam)
        config.update(**kwargs)
        return optimizers.Adam(**config)
    return method
