import os
import numpy as np
import pandas as pd

from typing import Dict, \
                   Optional, \
                   Union
from ct.preprocess.tokenize import Tokenizer


def wma(dataset: pd.DataFrame,
        tokenizer: Optional[Union[str, Tokenizer]] = None,
        tokenizer_input_paths: Optional[Dict[str, str]] = None,
        tokenizer_output_path: Optional[str] = None,
        train_val_split=0.8,
        inplace=True,
        vocab_size=30000,
        language='english',
        lowercase=False):
    assert tokenizer is not None or tokenizer_input_paths is not None, \
        'provide either a tokenizer, a tokenizer path, ' \
        'or `tokenizer input paths` to use for creating a new tokenizer.'
    if tokenizer is None:
        tokenizer = Tokenizer(input_paths=list(tokenizer_input_paths.values()),
                              tokenizer_output_path=tokenizer_output_path,
                              vocab_size=vocab_size,
                              lowercase=lowercase)
    elif isinstance(tokenizer, str):
        tokenizer = Tokenizer.load(path=tokenizer)

    dataset, train, val = _wma(dataset=dataset,
                               tokenizer=tokenizer,
                               train_val_split=train_val_split,
                               language=language,
                               inplace=inplace)

    base_dir = os.path.join(os.path.split(os.path.split(tokenizer_input_paths['en'])[0])[0])
    base_filename = f'en-v{vocab_size}-{"lowercase-" if lowercase else ""}-p{len(tokenizer_input_paths)}.pkl.zip'
    os.makedirs(os.path.join(base_dir, 'tokenized'), exist_ok=True)

    np.save(os.path.join(base_dir, 'tokenized', 'train-' + base_filename), train)
    np.save(os.path.join(base_dir, 'tokenized', 'val-' + base_filename), val)

    return tokenizer, dataset, train, val


def _wma(dataset,
         tokenizer,
         train_val_split=0.8,
         language='english',
         inplace=True):
    if not inplace:
        dataset = dataset.copy()

    column_ids = f'{language}_ids'
    encodings = tokenizer.encode_batch(dataset[language].tolist())
    dataset[column_ids] = [encoding.ids for encoding in encodings]

    val_index = int(len(dataset) * train_val_split)
    x_train = dataset[column_ids][:val_index]
    x_val = dataset[column_ids][-val_index:]

    x_train = np.array([ids for english_ids in x_train for ids in english_ids])
    x_val = np.array([ids for english_ids in x_val for ids in english_ids])
    return dataset, x_train, x_val
