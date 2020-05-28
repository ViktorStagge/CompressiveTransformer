from keras import optimizers
from omegaconf import OmegaConf
from dataclasses import dataclass


@dataclass
class Adam:
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999


def get_optimizer(method, **kwargs):
    if isinstance(method, str):
        method = method.lower()

    if method in ['adam']:
        conf = OmegaConf.structured(Adam)
        conf.update(**kwargs)
        return optimizers.Adam(**conf)
    return method
