import os
import pandas as pd


def _load_file(path):
    with open(path, 'r', encoding='utf8') as file:
        content = file.readlines()
    return content


def wma(path_first_language,
        path_second_language,
        first_language='english',
        second_language='german',
        output_path=None):
    first_sentences = _load_file(path_first_language)
    second_sentences = _load_file(path_second_language)

    data = pd.DataFrame(data={first_language: first_sentences,
                              second_language: second_sentences})

    if output_path is not None:
        data.to_pickle(output_path)

    return data
