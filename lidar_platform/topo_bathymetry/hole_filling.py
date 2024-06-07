import struct

import numpy as np
from osgeo import gdal
from osgeo import osr


def print_dataset_information(dataset):
    print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                 dataset.GetDriver().LongName))
    print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                        dataset.RasterYSize,
                                        dataset.RasterCount))
    print("Projection is {}".format(dataset.GetProjection()))
    geotransform = dataset.GetGeoTransform()
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))


def print_raster_band_information(dataset, band_index=1):
    band = dataset.GetRasterBand(band_index)
    print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

    val_min = band.GetMinimum()
    val_max = band.GetMaximum()
    if not val_min or not val_max:
        (val_min, val_max) = band.ComputeRasterMinMax(True)
    print("Min={:.3f}, Max={:.3f}".format(val_min, val_max))

    if band.GetOverviewCount() > 0:
        print("Band has {} overviews".format(band.GetOverviewCount()))

    if band.GetRasterColorTable():
        print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))


def read_raster_data(band):
    scanline = band.ReadRaster(xoff=0, yoff=0,
                               xsize=band.XSize, ysize=1,
                               buf_xsize=band.XSize, buf_ysize=1,
                               buf_type=gdal.GDT_Float32)
    tuple_of_floats = struct.unpack('f' * band.XSize, scanline)


def create(dst_filename, fmt='GTiff'):
    driver = gdal.GetDriverByName(fmt)
    dst_ds = driver.Create(dst_filename, xsize=512, ysize=512,
                           bands=1, eType=gdal.GDT_Byte)
    dst_ds.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    srs = osr.SpatialReference()
    srs.SetUTM(11, 1)
    srs.SetWellKnownGeogCS("NAD27")
    dst_ds.SetProjection(srs.ExportToWkt())
    raster = np.zeros((512, 512), dtype=np.uint8)
    dst_ds.GetRasterBand(1).WriteArray(raster)
    # Once we're done, close properly the dataset
    dst_ds = None


def create_copy(src, dst, array, no_data=None, no_data_value=-9999, format='GTiff'):
    driver = gdal.GetDriverByName(format)
    src_ds = gdal.Open(src)
    dst_ds = driver.CreateCopy(dst, src_ds, strict=0)
    if no_data is not None:
        array = ((1 * (no_data == 0)) * array) + (no_data_value * no_data)
    dst_ds.GetRasterBand(1).WriteArray(array)
    # Once we're done, close properly the dataset
    dst_ds = None
    src_ds = None


def check_if_driver_supports_create_and_create_copy(file_format):
    driver = gdal.GetDriverByName(file_format)
    metadata = driver.GetMetadata()
    if metadata.get(gdal.DCAP_CREATE) == "YES":
        print("Driver {} supports Create() method.".format(file_format))

    if metadata.get(gdal.DCAP_CREATECOPY) == "YES":
        print("Driver {} supports CreateCopy() method.".format(file_format))


def get_array_and_no_data_values(ds, n_band_id=1, no_data_value=-9999):
    band = ds.GetRasterBand(n_band_id)
    array = band.ReadAsArray()
    no_data_values = 1 * (array == no_data_value)
    return array, no_data_values


def fill_holes(raster_density, raster_dem, out, no_data_value):
    # open the density raster
    ds = gdal.Open(raster_density, gdal.GA_ReadOnly)
    density, no_data_density = get_array_and_no_data_values(ds, n_band_id=1, no_data_value=no_data_value)

    # open the DEM raster
    ds = gdal.Open(raster_dem, gdal.GA_ReadOnly)
    dem, no_data_dem = get_array_and_no_data_values(ds, n_band_id=1, no_data_value=no_data_value)

    # fill the holes where there is no density data but there is dem data
    select = (no_data_density == 1) & (no_data_dem == 0)
    density[select] = 0

    create_copy(raster_density, out, density,
                no_data=no_data_dem, no_data_value=no_data_value)
