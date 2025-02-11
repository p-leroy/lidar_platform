"""
Python package for point clouds data processing
"""

from .tools import cc, gdal, las, las_fmt, misc, sbf, m3c2
from .topo_bathymetry import bathymetry, sbet, water_surface
from .config import global_shifts
from .config.config import cc_std, cc_exe, cc_custom

__title__ = "lidar_platform"
__author__ = "Paul Leroy, Baptiste Feldmann, Mathilde Letard"
__credits__ = ["Paul Leroy", "Dimitri Lague", "Baptiste Feldmann", "OSUR LiDAR Platform"]
__status__ = "Development"
__url__ = "https://www.lidar.univ-rennes.fr"
__email__ = "paul.leroy@univ-rennes.fr"
__version__ = "0.0.0"
