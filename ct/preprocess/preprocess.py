import os
import pickle
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tokenizers import ByteLevelBPETokenizer


separator_samples = '############<new_sample>############'


def preprocess(dataset,
               text_column='text',
               max_vocab=None,
               char_level=False):
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


def tokenize(input_paths=None,
             input_dir=None,
             vocab_file=None,
             lowercase=False,
             dropout=None,
             vocab_size=30000,
             min_frequency=2,
             special_tokens=[],
             output_path=None):
    raise DeprecationWarning('use preprocess.Tokenizer instead.')
    assert (input_paths is not None) ^ (input_dir is not None), \
        'must specify either input_paths or input_dir to use for tokenization.'
    if input_dir:
        input_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]

    tokenizer = ByteLevelBPETokenizer(vocab_file=vocab_file,
                                      lowercase=lowercase,
                                      dropout=dropout)

    if vocab_file is None:
        tokenizer.train(files=input_paths,
                        vocab_size=vocab_size,
                        min_frequency=min_frequency,
                        special_tokens=special_tokens)

    if output_path:
        output_dir = os.path.join(*os.path.split(output_path)[:-1])
        name = os.path.split(output_path)[-1]
        tokenizer.save(output_dir, name=name)

    return tokenizer
