"""
Python package for point clouds data processing
"""

from . import classification, config, fwf, qc, scripts_bfe, tools, topo_bathymetry
from .tools import cc, las, misc
from .topo_bathymetry import bathymetry, sbet, water_surface
from .config import global_shifts

__title__ = "lidar_platform"
__author__ = "Baptiste Feldmann, Paul Leroy"
__credits__ = ["Baptiste Feldmann", "Dimitri Lague", "Paul Leroy", "OSUR LiDAR Platform"]
__status__ = "Development"
__url__ = "https://www.lidar-nantes-rennes.eu"
__email__ = "paul.leroy@univ-rennes1.fr"
__version__ = "0.0.0"
