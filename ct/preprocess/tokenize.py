import os
import yaml
import pickle
import warnings
import numpy as np
import pandas as pd

from typing import Dict, \
                   Optional
from omegaconf import OmegaConf
from tokenizers import ByteLevelBPETokenizer


separator_samples = '############<new_sample>############'


def keras_tokenizer(dataset,
                    text_column='text',
                    max_vocab=None,
                    char_level=False):
    from keras.preprocessing.text import Tokenizer
    warnings.warn('using keras\' tokenizer instead of default `Tokenizer`. '
                  'Results may worsen.', DeprecationWarning)

    tokenizer = Tokenizer(num_words=max_vocab,
                          lower=True,
                          char_level=char_level)
    tokenizer.fit_on_texts(dataset[text_column])
    processed = tokenizer.texts_to_sequences(dataset[text_column])
    processed = pd.Series(data=processed, index=dataset.index)
    return processed


class Tokenizer(ByteLevelBPETokenizer):
    def __init__(self,
                 input_paths=None,
                 input_dir=None,
                 vocab_file=None,
                 merges_file=None,
                 lowercase=False,
                 dropout=None,
                 vocab_size=30000,
                 min_frequency=2,
                 special_tokens=None,
                 tokens_output_dir=None,
                 tokenizer_output_path=None):
        super().__init__(vocab_file=vocab_file,
                         merges_file=merges_file,
                         lowercase=lowercase,
                         dropout=dropout)

        assert (input_paths is not None) ^ \
               (input_dir is not None) ^ \
               (vocab_file is not None and merges_file is not None), \
            'must specify either input_paths, input_dir, or vocab_file & merges_file to use for tokenization.'
        if input_dir and input_paths is None:
            input_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
        if special_tokens is None:
            special_tokens = []

        self.config = OmegaConf.create(dict(
            lowercase=lowercase,
            dropout=dropout,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
        ))

        if input_paths:
            self.train(files=input_paths,
                       vocab_size=vocab_size,
                       min_frequency=min_frequency,
                       special_tokens=special_tokens)

        if tokenizer_output_path is not None:
            self.save(path=tokenizer_output_path)
        if tokens_output_dir is not None:
            self.encode_files(input_paths=input_paths,
                              tokens_output_dir=tokens_output_dir)

    def encode_files(self,
                     input_paths=None,
                     input_dir=None,
                     return_encodings=True,
                     tokens_output_dir=None):
        assert (input_paths is not None) ^ (input_dir is not None), \
            'must specify either input_paths or input_dir to use for tokenization.'
        if input_dir and input_paths is None:
            input_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]

        encodings = []
        for path in input_paths:
            with open(path) as file:
                text = file.read()
            encoding = self.encode(text)

            if tokens_output_dir is not None:
                filename = os.path.split(path)[-1]
                if filename.endswith('.txt'):
                    filename = filename[:-4]
                    filename += '.pkl'

                with open(os.path.join(tokens_output_dir, filename), 'wb') as file:
                    pickle.dump(encoding.ids, file)
            if return_encodings:
                encodings.append(encoding)

        if return_encodings:
            return encodings

    def save(self, path=None, directory=None, name=None):
        directory, name = _split_path(path=path, directory=directory, name=name)
        config_path = os.path.join(directory, f'{name if name else "tokenizer"}.yaml')

        # saves merges and vocab files
        super().save(directory=directory, name=name)

        # saves configuration file
        with open(config_path, 'w') as file:
            yaml.dump(self.config.pretty(), file, default_flow_style=True)

    @staticmethod
    def load(path=None, directory=None, name=None):
        directory, name = _split_path(path=path, directory=directory, name=name)
        vocab_path = os.path.join(directory, f'{name + "-" if name else ""}vocab.json')
        merges_path = os.path.join(directory, f'{name + "-" if name else ""}merges.txt')
        config_path = os.path.join(directory, f'{name if name else "tokenizer"}.yaml')

        config = OmegaConf.load(config_path)
        tokenizer = Tokenizer(vocab_file=vocab_path,
                              merges_file=merges_path,
                              **config)
        return tokenizer


def _split_path(path=None, directory=None, name=None):
    assert (path is not None) ^ (directory is not None), \
        'must specify either output path or output directory.'
    if path:
        directory = os.path.join(*os.path.split(path)[:-1])
        name = os.path.split(path)[-1]
    return directory, name
