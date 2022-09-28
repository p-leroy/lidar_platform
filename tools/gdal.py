# coding: utf-8
# Paul Leroy
# Baptiste Feldmann

"""GDAL package
There are two ways to get gdal, the recommended way being to use conda: https://gdal.org/download.html#
It is also possible to use OSGeo4W and to add 'osgeo4w' to the path
https://trac.osgeo.org/osgeo4w/
The following scripts_bfe makes external calls to gdalbuildvrt, gdal_calc and gdal_merge
with gdal_calc, be careful with expressions like "A<0" as the < character can be misinterpreted by the subprocess
"""

import time
import os
import sys

from plateforme_lidar import utils


gdal_bin = "C:\\Users\\pleroy\\miniconda3\\Library\\bin"
gdal_scripts = "C:\Users\pleroy\miniconda3\Scripts"
gdalbuilddvrt = os.path.join(gdal_bin, "gdalbuildvrt")
gdal_calc = os.path.join(gdal_scripts, "gdal_cal.py")


def _exception(filepath):
    if not os.path.exists(filepath):
        sys.exit(os.path.split(filepath)[1] + " doesn't exists !")


def build_vrt(filepath, nodata=-9999):
    """Gdal build VRT, VRT = GDAL Virtual Format

    Args:
        filepath (str): path to raster file
        nodata (int, optional): if you want to modify No data value. Defaults to -9999.
    """
    print("[GDAL] build VRT...", end=" ")
    begin = time.time()
    buildvrt_input = filepath[0:-4] + "_buildvrtInputfile.txt"
    f = open(buildvrt_input, 'w')
    f.write(filepath)
    f.close()
    out = filepath[0:-4] + '_virtual.vrt'
    # gdalbuildvrt builds a VRT (Virtual) Dataset from a list of datasets.
    # In case the resolution of all input files is not the same,
    # -resolution
    # enables the user to control the way the output resolution is computed.
    # average is the default and will compute an average of pixel dimensions within the set of source rasters.
    # -r
    # Select a resampling algorithm. {nearest (default),bilinear,cubic,cubicspline,lanczos,average,mode}

    query = f"{gdalbuilddvrt} -resolution average -r nearest" + \
                  ' -srcnodata "' + str(nodata) + '" -input_file_list ' + buildvrt_input + ' ' + out
    os.system(query)

    os.remove(buildvrt_input)
    print("done in %.1f sec" % (time.time() - begin))

    return out


def raster_calc(expression, out_file, file_a, *args):
    """Gdal_calc

    Args:
        expression (str): ex:'A+B'
        out_file (str): output file path
        file_a (str): input file path A
        *args (str): extra input path (B,C,D,...)
    """
    print("[GDAL] raster calc...", end=" ")
    begin = time.time()
    # be careful with the calc option, quotes, no quotes, special characters as <... there are traps
    # especially in the case calc='A<0', the < character can be misinterpreted by the subprocess module
    # when launching the command (interpreted as an input redirect)
    query = gdal_calc + ' --calc "' + expression + '"' + \
            ' --format GTiff --type Float32 --NoDataValue -9999.0 -A ' + \
            file_a + ' --A_band 1' + " --outfile " + out_file
    for i in range(0, len(args)):
        letter = chr(66 + i)
        query += ' -' + letter + ' ' + args[i] + ' --' + letter + '_band=1'
    os.system(query)
    print("done in %.1f sec" % (time.time() - begin))

    return query


def merge(files, out_file):
    """Gdal_merge

    Args:
        files (list): list of input files
        out_file (str): output file path
    """
    print("[gdal.merge] merge tif files", end=" ")
    begin = time.time()
    f = open(out_file[0:-4] + "_mergeInputFiles.txt", 'w')
    for i in files:
        f.write(os.path.abspath(i) + "\n")
    f.close()
    query = "gdal_merge -n -9999 -a_nodata -9999 -ot Float32 -of GTiff -o " \
            + out_file + " --optfile " + out_file[0:-4] + "_mergeInputFiles.txt"
    utils.run(utils.GDAL_QUERY_ROOT + query, True, opt_shell=True)
    os.remove(out_file[0:-4] + "_mergeInputFiles.txt")
    print("done in %.1f sec" % (time.time() - begin))


def hole_filling(raster_density, raster_dem, debug=False):
    """Script to fill holes in density raster
    fill pixels with 0 when they are in the area of the DEM

    Args:
        raster_density (str): density raster path
        raster_dem (str): DEM raster path
    """

    print("[gdal.hole_filling] this function is deprecated, use fill_holes instead")

    head, tail = os.path.split(raster_density)

    raster_density_vrt = raster_density[0:-4] + "_virtual.vrt"
    raster_density_mask1 = raster_density[0:-4] + "_mask1.tif"
    raster_density_mask2 = raster_density[0:-4] + "_mask2.tif"
    raster_density_mask3 = raster_density[0:-4] + "_mask3.tif"
    raster_dem_mask1 = raster_dem[0:-4] + "_mask1.tif"

    # build a VRT (Virtual Dataset) using the density raster
    build_vrt(raster_density, -99)
    _exception(raster_density_vrt)

    raster_calc("A<0", raster_density_mask1, raster_density_vrt)
    _exception(raster_density_mask1)

    raster_calc("A>-9999", raster_dem_mask1, raster_dem)
    _exception(raster_dem_mask1)

    raster_calc("logical_and(A,B)",
                raster_density_mask2,
                raster_density_mask1,
                raster_dem_mask1)
    _exception(raster_density_mask2)

    raster_calc("(A*9999)-9999", raster_density_mask3, raster_density_mask2)
    _exception(raster_density_mask3)

    out = os.path.join(head, 'final', tail)
    merge([raster_density, raster_density_mask3], out)

    if not debug:
        os.remove(raster_density_vrt)
        os.remove(raster_density_mask1)
        os.remove(raster_density_mask2)
        os.remove(raster_density_mask3)
        os.remove(raster_dem_mask1)

    return out
