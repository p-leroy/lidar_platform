# coding: utf-8
# Baptiste Feldmann

"""GDAL package
Need to have downloaded OSGEO4W and added 'osgeo4w' in environment variable
See more : https://trac.osgeo.org/osgeo4w/
Because of complexity to install (and run) properly gdal in python environment, this script use OSGEO4W to interects with by command-line
"""

import time
import os
import subprocess
import sys

from . import utils


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

    # query = 'gdalbuildvrt -resolution average -r nearest -srcnodata "' + str(nodata) + \
    #         '" -input_file_list ' + buildvrt_input + ' ' + out
    # utils.run(utils.GDAL_QUERY_ROOT + query, True, opt_shell=True)

    other_query = "C:\\Users\\pleroy\\miniconda3\\Library\\bin\\gdalbuildvrt -resolution average -r nearest" + \
                  ' -srcnodata "' + str(nodata) + '" -input_file_list ' + buildvrt_input + ' ' + out
    os.system(other_query)

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
    # NO QUOTES AROUND THE CALC OPTION!!! AT LEAST IN JUPYTER NOTEBOOKS
    query = 'osgeo4w gdal_calc --format=GTiff --type=Float32 --NoDataValue=-9999.0 -A ' + \
            file_a + ' --A_band=1'
    for i in range(0, len(args)):
        letter = chr(66 + i)
        query += ' -' + letter + ' ' + args[i] + ' --' + letter + '_band=1'
    query += " --outfile " + out_file + ' --calc=' + expression
    # utils.run(query, True, opt_shell=True)
    #os.system(query)
    other_query = 'C:\\Users\\pleroy\\miniconda3\\Scripts\\gdal_calc.py' + \
                  ' --calc "' + expression + '"' + \
                  ' --format GTiff --type Float32 --NoDataValue -9999.0 -A ' + \
                  file_a + ' --A_band 1' + " --outfile " + out_file
    for i in range(0, len(args)):
        letter = chr(66 + i)
        other_query += ' -' + letter + ' ' + args[i] + ' --' + letter + '_band=1'
    os.system(other_query)
    #subprocess.call(other_query, shell=True)
    print("done in %.1f sec" % (time.time() - begin))

    return other_query


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


def hole_filling(raster_density, raster_dem):
    """Script to fill holes in density raster
    fill pixels with 0 when they are in the area of the DEM

    Args:
        raster_density (str): density raster path
        raster_dem (str): DEM raster path
    """

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

    os.remove(raster_density_vrt)
    os.remove(raster_density_mask1)
    os.remove(raster_density_mask2)
    os.remove(raster_density_mask3)
    os.remove(raster_dem_mask1)

    return out
