# coding: utf-8
# Baptiste Feldmann

"""GDAL package
Need to have downloaded OSGEO4W and added 'osgeo4w' in environment variable
See more : https://trac.osgeo.org/osgeo4w/
Because of complexity to install (and run) properly gdal in python environment, this script use OSGEO4W to interects with by command-line
"""

import time,os,sys
from . import utils

def _exception(filepath):
    if not os.path.exists(filepath):
        sys.exit(os.path.split(filepath)[1]+" doesn't exists !")
        
def Buildvrt(filepath,nodata=-9999):
    """Gdal build VRT

    Args:
        filepath (str): path to raster file
        nodata (int, optional): if you want to modify No data value. Defaults to -9999.
    """
    print("[GDAL] build VRT...",end=" ")
    begin=time.time()
    #splitname=os.path.split(filepath)
    f=open(filepath[0:-4]+"_buildvrtInputfile.txt",'w')
    f.write(filepath)
    f.close()
    query='gdalbuildvrt -resolution average -r nearest -srcnodata "'+str(nodata)+'" -input_file_list '+filepath[0:-4]+\
           '_buildvrtInputfile.txt '+filepath[0:-4]+'_virtual.vrt'
    utils.Run(utils.GDAL_QUERY_ROOT+query,True,optShell=True)
    os.remove(filepath[0:-4]+"_buildvrtInputfile.txt")
    print("done in %.1f sec" %(time.time()-begin))

def RasterCalc(expression,outFile,fileA,*args):
    """Gdal_calc

    Args:
        expression (str): ex:'A+B'
        outFile (str): output file path
        fileA (str): input file path A
        *args (str): extra input path (B,C,D,...)
    """
    print("[GDAL] raster calc...",end=" ")
    begin=time.time()
    query='gdal_calc --calc "'+expression+'" --format GTiff --type Float32 --NoDataValue -9999.0 -A '+fileA+' --A_band 1'
    for i in range(0,len(args)):
        letter=chr(66+i)
        query+=' -'+letter+' '+args[i]+' --'+letter+'_band 1'
    query+=" --outfile "+outFile
    utils.Run("osgeo4w & call "+query,True,optShell=True)
    print("done in %.1f sec" %(time.time()-begin))

def Merge(listFiles,outFile):
    """Gdal_merge

    Args:
        listFiles (list): list of input files
        outFile (str): output file path
    """
    print("[GDAL] raster merge...",end=" ")
    begin=time.time()
    f=open(outFile[0:-4]+"_mergeInputFiles.txt",'w')
    for i in listFiles:
        f.write(os.path.abspath(i)+"\n")
    f.close()
    query="gdal_merge -n -9999 -a_nodata -9999 -ot Float32 -of GTiff -o "+outFile+" --optfile "+outFile[0:-4]+"_mergeInputFiles.txt"
    utils.Run(utils.GDAL_QUERY_ROOT+query,True,optShell=True)
    os.remove(outFile[0:-4]+"_mergeInputFiles.txt")
    print("done in %.1f sec" %(time.time()-begin))

def HoleFilling(rasterDensity,rasterDEM):
    """Script to fill holes in density raster
    fill pixel with 0 value when pixels are in the area of DEM

    Args:
        rasterDensity (str): density raster path
        rasterDEM (str): DEM raster path
    """
    splitname=os.path.split(rasterDensity)
    Buildvrt(rasterDensity,-99)
    _exception(rasterDensity[0:-4]+"_virtual.vrt")
    
    RasterCalc('A<0',rasterDensity[0:-4]+"_mask1.tif",rasterDensity[0:-4]+"_virtual.vrt")
    _exception(rasterDensity[0:-4]+"_mask1.tif")
    
    RasterCalc("A>-9999",rasterDEM[0:-4]+"_mask1.tif",rasterDEM)
    _exception(rasterDEM[0:-4]+"_mask1.tif")
    
    RasterCalc("logical_and(A,B)",rasterDensity[0:-4]+"_mask2.tif",rasterDensity[0:-4]+"_mask1.tif",rasterDEM[0:-4]+"_mask1.tif")
    _exception(rasterDensity[0:-4]+"_mask2.tif")
    
    RasterCalc("(A*9999)-9999",rasterDensity[0:-4]+"_mask3.tif",rasterDensity[0:-4]+"_mask2.tif")
    _exception(rasterDensity[0:-4]+"_mask3.tif")
    
    Merge([rasterDensity,rasterDensity[0:-4]+"_mask3.tif"],splitname[0]+"/final/"+splitname[1])
    os.remove(rasterDensity[0:-4]+"_mask1.tif")
    os.remove(rasterDensity[0:-4]+"_mask2.tif")
    os.remove(rasterDensity[0:-4]+"_mask3.tif")
    os.remove(rasterDensity[0:-4]+"_virtual.vrt")
    os.remove(rasterDEM[0:-4]+"_mask1.tif")

