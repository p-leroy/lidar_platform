"""
Python package for topo-bathymetric airborne LiDAR processing
"""

from . import calculs, cc_3dmasc, classification, cloudcompare, lasfwf, lastools, PySBF, sbet, gdal, utils, test

__title__="plateforme_lidar"
__author__="Baptiste Feldmann"
__credits__=["Baptiste Feldmann","Dimitri Lague","Nantes-Rennes LiDAR research Platform"]
__status__="Development"
__url__="https://www.lidar-nantes-rennes.eu"
__email__="baptiste.feldmann@univ-rennes1.fr"
__version__="0.3.0"

test.check()
