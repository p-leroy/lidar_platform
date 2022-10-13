# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:28:53 2021

@author: PaulLeroy
"""

import os
import socket

import numpy as np

# LAStools binaries
bin_las = 'C:\\opt\\LAStools\\bin'

# CloudCompare paths
cc_std = '"C:\\Program Files\\CloudCompare\\CloudCompare.exe"'  # standard CloudCompare
cc_std_alt = cc_std[1:-1]  # this is for proper usage in subprocesses
# other CloudCompare versions on lidar-server
cc_2022_07_05 = r'G:\RENNES1\PaulLeroy\CloudCompare_2022_07_05\CloudCompare.exe'

# configure CloudCompare aliases
hostname = socket.gethostname()
if hostname == 'LIDAR-SERVER':
    cc_custom = cc_2022_07_05
elif hostname == 'DESKTOP-0T01J23' or hostname == 'DESKTOP-0L5SMT4':
    cc_custom = 'C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe'
else:
    cc_custom = None

# fill in the QUERY_0 dictionary for legacy calls to helper applications
if hostname == 'LIDAR-SERVER':
    QUERY_0 = {"standard" : cc_std + ' -silent',
               "standard_view" : cc_std,
               "PoissonRecon" : "G:/RENNES1/BaptisteFeldmann/AdaptiveSolvers/PoissonRecon.exe",
               "cc_ple" : "G:/RENNES1/BaptisteFeldmann/CloudCompare_PL_01042022/CloudCompare.exe -silent",
               "cc_ple_view" : "G:/RENNES1/BaptisteFeldmann/CloudCompare_PL_01042022/CloudCompare.exe"
               }
else:
    QUERY_0 = {"standard": cc_std + ' -silent',
               "standard_view": cc_std,
               "PoissonRecon": 'C:\opt\AdaptiveSolvers.x64\PoissonRecon.exe',
               "cc_ple": 'C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe -silent',
               "cc_ple_view": 'C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe'
               }

EXPORT_FMT = {"LAS" : " -c_export_fmt LAS -ext laz -auto_save OFF",
              "LAS_auto_save" : " -c_export_fmt LAS -ext laz",
              "BIN_auto_save" : " -c_export_fmt BIN",
              "SBF_auto_save" : " -c_export_fmt SBF",
              "PLY_cloud" : " -c_export_fmt PLY -PLY_export_fmt BINARY_LE -auto_save OFF",
              "PLY_mesh" : " -m_export_fmt PLY -PLY_export_fmt BINARY_LE -auto_save OFF",
              "SBF" : " -c_export_fmt SBF -auto_save OFF"}

## global shifts for CloudCompare
SHIFT = {}
globalShiftFile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "global_shift.txt")
for i in np.loadtxt(globalShiftFile, str, delimiter=";"):
    SHIFT[i[0]] = i[1]

# vertical datum folder
if hostname == 'LIDAR-SERVER':
    VERTICAL_DATUM_DIR = r'G:\RENNES1\BaptisteFeldmann\Vertical_datum'
else:
    VERTICAL_DATUM_DIR = r'C:\DATA\Vertical_datum'

# GDAL access using OSGeo4W
GDAL_QUERY_ROOT = "osgeo4w "
