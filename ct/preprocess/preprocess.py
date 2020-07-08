import os
import pickle
import numpy as np
import pandas as pd
from tokenizers import ByteLevelBPETokenizer


separator_samples = '############<new_sample>############'


def preprocess(dataset,
               text_column='text',
               max_vocab=None,
               char_level=False):
    from keras.preprocessing.text import Tokenizer

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
        if input_dir:
            raise NotImplementedError('appears to break and not converge for many [small] files.')
            input_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
        if special_tokens is None:
            special_tokens = []

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
                     tokens_output_dir=None):
        assert (input_paths is not None) ^ (input_dir is not None), \
            'must specify either input_paths or input_dir to use for tokenization.'

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
            else:
                encodings.append(encoding)

        if tokens_output_dir is None:
            return encodings

    def save(self, path=None, directory=None, name=None):
        assert (path is not None) ^ (directory is not None), \
            'must specify either output path or output directory.'

        if path:
            directory = os.path.join(*os.path.split(path)[:-1])
            name = os.path.split(path)[-1]
        super().save(directory=directory, name=name)

    @staticmethod
    def load(path=None, directory=None, name=None):
        assert (path is not None) ^ (directory is not None), \
            'must specify either output path or output directory.'

        if path:
            directory = os.path.join(*os.path.split(path)[:-1])
            name = os.path.split(path)[-1]
        raise NotImplementedError


def preprocess_wma(input_paths,
                   dataset,
                   tokenizer_output_path,
                   train_val_split=0.8,
                   inplace=True,
                   vocab_size=30000,
                   language='english',
                   lowercase=False,
                   tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(input_paths=list(input_paths.values()),
                              tokenizer_output_path=tokenizer_output_path,
                              vocab_size=vocab_size,
                              lowercase=lowercase)
    elif isinstance(tokenizer, str):
        tokenizer = Tokenizer.load(path=tokenizer)
    
    dataset, x_train, x_val = _preprocess_wma(dataset=dataset,
                                              tokenizer=tokenizer,
                                              train_val_split=train_val_split,
                                              language=language,
                                              inplace=inplace)

    base_dir = os.path.join(*os.path.split(input_paths['en'])[:-2])
    base_filename = f'en-v{vocab_size}-{"lowercase-" if lowercase else ""}-p{len(input_paths)}.pkl.zip'
    os.makedirs(os.path.join(base_dir, 'tokenized'), exist_ok=True)

    x_train.to_pickle(os.path.join(base_dir, 'tokenized', 'train-' + base_filename))
    x_val.to_pickle(os.path.join(base_dir, 'tokenized', 'val-' + base_filename))
    
    return dataset, x_train, x_val


def _preprocess_wma(dataset,
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
    x_train = dataset[[column_ids]][:val_index]
    x_val = dataset[[column_ids]][-val_index:]
    
    x_train = np.array([ids for english_ids in x_train for ids in english_ids])
    x_val = np.array([ids for english_ids in x_val for ids in english_ids])
    return dataset, x_train, x_val
