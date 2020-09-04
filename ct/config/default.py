import os

from typing import Dict, \
                   Any, \
                   Optional, \
                   List
from dataclasses import dataclass, \
                        field
from omegaconf import OmegaConf


@dataclass
class _Config:
    # ## Meta ### #
    dataset: str = 'pg-19'
    version: int = 2
    verbose: bool = True
    feature_relative_encoding: bool = False

    # ## Run ### #
    tokenize: bool = False
    train: bool = True

    # ## Tokenize ### #
    save_tokens: bool = True
    load_tokens: bool = True
    lowercase: bool = False
    vocab_size: int = 16384
    max_tokens_files: Optional[int] = None

    # ### Training ### #
    continue_training: bool = True
    train_steps: int = 12000000
    validation_steps: int = 100000

    epochs: int = 5
    batch_size: int = 128
    d_layers: int = 2
    d_heads: int = 2
    sequence_length: int = 128
    memory_size: int = 256
    compressed_memory_size: int = 256
    d_model: int = 128
    d_k: int = 16
    output_size: int = vocab_size
    steps_per_epoch: int = train_steps//sequence_length - 1
    save_interval: int = 5000

    # ### Paths ### #
    input_dir: str = 'data/deepmind-gutenberg/input/train/'
    input_paths: List[str] = field(default_factory=list)
    tokenizer_output_path: str = f'data/deepmind-gutenberg/tokenizer/pg19-' \
                                 f't{vocab_size}-' \
                                 f'v{version}.tok'
    tokens_output_dir: str = f'data/deepmind-gutenberg/tokenized/v{version}'
    processed_path: str = f'data/deepmind-gutenberg/processed/v{version}/train.pkl'
    train_logs_output_path: str = f'training-logs/ct-pg19-v{version}.txt'
    model_output_path: str = f'data/deepmind-gutenberg/model/' \
                             f'test-ct-pg-19-' \
                             f'v{version}-' \
                             f'e{epochs}-' \
                             f'vs{vocab_size}-' \
                             f'bs{batch_size}-' \
                             f'l{d_layers}-' \
                             f's{sequence_length}-' \
                             f'd{d_model}.h5'


config = OmegaConf.structured(_Config)
config.input_paths = [os.path.join(config.input_dir, filename) for filename in os.listdir(config.input_dir)]
