
import click
import rasterio as rio
from color_balance import *

# Ugh. Make consistent naming convention for module and files.
from color_balance.colorimage import get_histogram


@click.command('color-balance')
@click.argument('srcpath', type=click.Path(exists=True))
@click.argument('refpath', type=click.Path(exists=True))
@click.argument('dstpath', type=click.Path(exists=False))
def color_balance(srcpath, refpath, dstpath):
    """
    Apply color information from a reference image to a source image.
    """

    with rio.open(refpath) as ref, rio.open(srcpath) as src:
        
        count = src.count
        assert ref.count == count
        
        for bidx in range(1, count + 1):
            
            # TODO: No need to store both bands in memory, even temporarily.
            reference = ref.read(bidx)
            source = src.read(bidx)
            
            ref_hist = get_histogram(reference)
            src_hist = get_histogram(source)
            
            
            