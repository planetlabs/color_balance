import numpy
import cv2

from osgeo import gdal, gdal_array
from color_balance import colorimage as ci


def load_mask(mask_path, conservative=True):
    '''Loads a single-band mask image and adds it to the mask array.
    If conservative is True, only totally-masked pixels are masked out.
    Otherwise, all partially pixels are masked out.'''
    gdal_ds = gdal.Open(mask_path)
    if gdal_ds is None:
        raise IOError('Mask could not be opened.')

    band = gdal_ds.GetRasterBand(1)
    mask_array = band.ReadAsArray()

    if conservative:
        # Partially-masked pixels set to unmasked
        mask_array[mask_array > 0] = 255
    else:
        # Partially-masked pixels set to masked
        mask_array[mask_array < 255] = 0

    return mask_array


def load_image(image_path, band_indices=None, bit_depth=None,
               curve_function=None):
    '''Loads an image into a colorimage. If no bit depth is supplied, 8-bit
    is assumed. If no band indices are supplied, RGB is assumed.'''
    raster = CImage()
    raster.load(filename=image_path)
    img, mask = ci.convert_to_colorimage(raster, band_indices=band_indices,
                                         bit_depth=bit_depth,
                                         curve_function=curve_function)
    return img, mask, raster


def save_adjusted_image(filename, img, mask, cimage):
    '''Saves the colorimage to a new raster, writing the mask as the alpha
    channel and copying the geographic information from the original raster'''
    cimage.alpha = mask
    for i, band in enumerate(reversed(cv2.split(img))):
        cimage.bands[i] = band.astype(numpy.uint8)
    cimage.save(filename)


class CImage:
    '''Make IO easier'''

    def __init__(self, bands=[], alpha=None, metadata={}):
        self.bands = bands

        if alpha is None:
            self.create_alpha()
        else:
            self.alpha = alpha

        self.metadata = metadata

    def create_alpha(self):
        shape = self.bands[0].shape
        self.alpha = 255*numpy.ones(shape, dtype=numpy.uint8)

    def load(self, filename):
        gdal_ds = gdal.Open(filename)
        if gdal_ds is None:
            raise Exception('Unable to open file "{}" with gdal.Open()'.format(
                filename))

        # Read alpha band
        alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
        if alpha_band.GetColorInterpretation() == gdal.GCI_AlphaBand:
            self.alpha = alpha_band.ReadAsArray()
            band_count = gdal_ds.RasterCount - 1
        else:
            band_count = gdal_ds.RasterCount

        # 16bit TIFF files have 16bit alpha, adjust to 8bit.
        if self.alpha is not None \
                and gdal_ds.GetRasterBand(1).DataType == gdal.GDT_UInt16 \
                and numpy.max(self.alpha) > 255:
            self.alpha = (self.alpha / 256).astype(numpy.uint8)

        # Read raster bands
        for band_n in range(1, band_count+1):
            band = gdal_ds.GetRasterBand(band_n)
            array = band.ReadAsArray()
            if array is None:
                raise Exception('GDAL error occured : {}'.format(
                    gdal.GetLastErrorMsg()))
            self.bands.append(array)

        # Load georeferencing information
        self.metadata['geotransform'] = gdal_ds.GetGeoTransform()
        self.metadata['projection'] = gdal_ds.GetProjection()
        self.metadata['rpc'] = gdal_ds.GetMetadata('RPC')


    def save(self, filename, options = []):
        options = list(options)
        
        band_count = len(self.bands)
        ysize, xsize = self.bands[0].shape

        if band_count == 3:
            options.append('PHOTOMETRIC=RGB')
        
        if self.alpha is not None:
            band_count += 1
            options.append('ALPHA=YES')
        
        datatype = gdal_array.NumericTypeCodeToGDALTypeCode( 
            self.bands[0].dtype.type )
        
        gdal_ds = gdal.GetDriverByName('GTIFF').Create(
            filename, xsize, ysize, band_count, datatype,
            options = options)

        # Save georeferencing information
        if 'projection' in self.metadata.keys():
            gdal_ds.SetProjection(self.metadata['projection'])
        if 'geotransform' in self.metadata.keys():
            gdal_ds.SetGeoTransform(self.metadata['geotransform'])
        if 'rpc' in self.metadata.keys():
            gdal_ds.SetMetadata(self.metadata['rpc'], 'RPC')

        for i in range(len(self.bands)):
            gdal_array.BandWriteArray(
                gdal_ds.GetRasterBand(i+1),
                self.bands[i])
            
        if self.alpha is not None:
            alpha = self.alpha
            alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)

            # To conform to 16 bit TIFF alpha expectations transform 
            # alpha to 16bit.
            if alpha_band.DataType == gdal.GDT_UInt16:
                alpha = ((alpha.astype(numpy.uint32) * 65535) / 255).astype(
                    numpy.uint16)
