from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class DefaultConfig:
    feature_relative_encoding: bool = False


default_config = OmegaConf.structured(DefaultConfig)
