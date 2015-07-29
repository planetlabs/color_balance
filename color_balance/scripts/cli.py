
import click
from color_balance import *


@click.command('color-balance')
@click.argument('srcpath', type=click.Path(exists=True))
@click.argument('refpath', type=click.Path(exists=True))
@click.argument('dstpath', type=click.Path(exists=False))
def color_balance():
    """
    Apply color information from a reference image to a source image.
    """
    click.echo('Not Implemented Yet')