from omegaconf import OmegaConf

from ct.config.default import config


_configs = OmegaConf.create(dict(default=config))


def get_config(path='default', **kwargs):
    if path in _configs:
        return _configs[path]

    config = OmegaConf.load(path)
    _configs[path] = config
    return config
