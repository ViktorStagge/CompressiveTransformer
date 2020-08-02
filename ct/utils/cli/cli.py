import click
from train import train as _train
from preprocess import Tokenizer
from load.wma import wma as load_wma


@click.group()
def main_group(**kwargs):
    pass


@main_group.command()
@click.option('--dataset',
              help='specify the dataset to use',
              default='treebank',
              show_default=True)
def train(**kwargs):
    _train(**kwargs)


@main_group.command()
@click.option('--input-path',
              'input_paths',
              help='specify the input files to use for creating the tokenizer',
              required=True,
              multiple=True,
              prompt=True)
@click.option('--output-dir',
              'output_dir',
              default=None,
              help='(optional) specify the output directory for the created tokens')
@click.option('--tokenizer-output-path',
              'tokenizer_output_path',
              default=None,
              help='(optional) specify the output file for the created tokenizer')
def tokenize(input_paths, output_dir, tokenizer_output_path, **kwargs):
    assert isinstance(input_paths, tuple), \
        'unexpected format for input_paths. Expected `multiple` to be set to True.'
    input_paths = list(input_paths)

    print(f'Using input files: {input_paths}')
    print(f'Using kwargs: {kwargs}')
    if output_dir:
        print(f'Saving tokens: {output_dir}')
    if tokenizer_output_path:
        print(f'Saving tokenizer: {tokenizer_output_path}')

    Tokenizer(input_paths,
              tokens_output_dir=output_dir,
              tokenizer_output_path=tokenizer_output_path)


@main_group.command()
@click.option('--input-dir',
              'input_dir',
              help='directory containing input files',
              required=True,
              prompt=True)
@click.option('--tokenizer-output-path',
              'tokenizer_output_path',
              help='output path for generated tokenizer',
              required=True,
              prompt=True)
@click.option('--tokens-output-dir',
              'tokens_output_dir',
              help='output directory for generated, tokenized files',
              required=True,
              prompt=True)
@click.option('--output-path',
              'processed_path',
              help='final output path for the preprocessed data',
              required=True,
              prompt=True)
@click.option('--input-paths',
              'input_paths',
              help='(optional) explicitly specify which input files to use',
              multiple=True)
@click.option('--vocab-size',
              'vocab_size',
              default=30000,
              help='(optional) maximum vocabulary size used for tokenizer')
@click.option('--lowercase',
              'lowercase',
              default=False,
              help='(optional) use only lowercase characters for tokenizer')
def preprocess(**kwargs):
    from preprocess.dataset import preprocess
    preprocess(**kwargs)


@main_group.group()
@click.pass_context
def reformat(*args, **kwargs):
    pass


@reformat.command()
@click.option('--path-first-language',
              'path_first_language',
              help='input file containig sentences from one of the two languages',
              required=True,
              prompt=True)
@click.option('--path-second-language',
              'path_second_language',
              help='input file containig sentences from one of the two languages',
              required=True,
              prompt=True)
@click.option('--first-language',
              'path_second_language',
              help='(optional) name of first language',
              show_default=True,
              default='english')
@click.option('--second-language',
              'path_second_language',
              help='(optional) name of second language',
              show_default=True,
              default='german')
@click.option('--output-path',
              'output_path',
              required=True,
              help='output path for the created DataFrame')
def wma(**kwargs):
    load_wma(**kwargs)


# main_group.add_command(reformat)
