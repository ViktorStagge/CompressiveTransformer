import os
import pickle
import numpy as np

from typing import Optional, \
                   List, \
                   Tuple

from preprocess import Tokenizer


def preprocess(input_dir: str,
               tokenizer_output_path: str,
               tokens_output_dir: str,
               processed_path: str,
               input_paths: Optional[List] = None,
               vocab_size: int = 30000,
               lowercase: bool = False) \
        -> Tuple[Tokenizer, list]:

    tokenizer = Tokenizer(input_dir=input_dir,
                          input_paths=input_paths,
                          tokenizer_output_path=tokenizer_output_path,
                          tokens_output_dir=tokens_output_dir,
                          vocab_size=vocab_size,
                          lowercase=lowercase)

    token_paths = _list_all_paths(tokens_output_dir)
    tokens = [_load_pickle(path) for path in token_paths]
    processed = np.array(_flatten(tokens))

    np.save(processed_path, processed)
    # _save_file(processed_path, processed)

    return tokenizer, processed


def _load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


# def save_file(path, content):
#     with open(path, 'wb') as file:
#         pickle.dump(content, file)


def _flatten(nested_list):
    return [t for document in nested_list for t in document]


def _list_all_paths(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory)]
