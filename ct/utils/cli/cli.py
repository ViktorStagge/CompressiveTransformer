import click
from train import train as _train
from preprocess import tokenize as _tokenize

@click.group()
def main_group(**kwargs):
    pass


@click.command()
@click.option('--dataset',
              help='specify the dataset to use',
              default='treebank',
              show_default=True)
def train(**kwargs):
    _train(**kwargs)


@click.command()
@click.option('--input_path',
              'input_paths',
              help='specify the input files to use for creating the tokenizer',
              required=True,
              multiple=True,
              prompt=True)
@click.option('--output_path',
              'output_path',
              default=None,
              help='(optional) specify the output file for the created tokenizer')
def tokenize(input_paths, output_path, **kwargs):
    assert isinstance(input_paths, tuple), \
        'unexpected format for input_paths. Expected `multiple` to be set to True.'
    input_paths = list(input_paths)
    print(f'Using input files: {input_paths}')
    print(f'Using kwargs: {kwargs}')
    if output_path:
        print(f'Saving tokenizer: {output_path}')

    _tokenize(input_paths, output_path=output_path)


main_group.add_command(train)
main_group.add_command(tokenize)
