
import click
import rasterio as rio

# Make consistent naming convention for module and files.
# e.g. colorimage -> color_image or color_balance -> color_balance
# or just use shorter names.
from color_balance.colorimage import get_cdf, apply_lut
from color_balance.histogram_match import cdf_match_lut


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

        profile = src.profile

        with rio.open(dstpath, 'w', **profile) as dst:

            for bidx in range(1, count + 1):

                # TODO: No need to store both bands in memory, even temporarily.
                source = src.read(bidx)
                reference = ref.read(bidx)

                src_cdf = get_cdf(source)
                ref_cdf = get_cdf(reference)

                lut = cdf_match_lut(src_cdf, ref_cdf, dtype=source.dtype)
                arr = apply_lut(source, lut)

                dst.write_band(bidx, arr)