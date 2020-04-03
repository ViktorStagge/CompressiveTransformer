import os
import re
import pandas as pd


def treebank(path_to_dir):
    assert os.path.exists(path_to_dir), f'could not find dir "{path_to_dir}"'

    filenames = [filename for filename in os.listdir(path_to_dir) if filename != 'README']
    filepaths = [os.path.join(path_to_dir, filename) for filename in filenames]

    data = []
    for path, filename in zip(filepaths, filenames):
        with open(path) as file:
            text = file.read()
        assert text.startswith('.START'), f'unknown format for: "{text[:20]}..."'
        text = text[6:].strip()
        data.append(text)

    data = pd.DataFrame(data=data,
                        columns=['text'],
                        index=filenames)
    return data
