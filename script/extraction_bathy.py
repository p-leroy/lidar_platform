# coding: utf-8
# Baptiste Feldmann

import glob, os

from joblib import Parallel, delayed
import numpy as np

import plateforme_lidar as pl


def extract_bathy(filename, filter_v=[0.25, -10], filter_xy=250):
    head, tail = os.path.split(filename)
    odir = os.path.join(head, "extraction")
    in_data = pl.lastools.readLAS_laspy(filename, extraField=True)
    dist_xy = np.sqrt((in_data['c2c_absolute_distances_x'] ** 2) + (in_data['c2c_absolute_distances_y'] ** 2))

    select = (in_data['c2c_absolute_distances_z'] < filter_v[0]) & (in_data['c2c_absolute_distances_z'] > filter_v[1])
    select &= (dist_xy < filter_xy)
    in_data_bathy = pl.lastools.Filter_LAS(in_data, select)
    in_data_topo = pl.lastools.Filter_LAS(in_data, np.logical_not(select))
    
    extra = [(("depth", "float32"), np.round(in_data_bathy['c2c_absolute_distances_z'], decimals=2))]
    if len(in_data_bathy) > 1:
        pl.lastools.WriteLAS(os.path.join(odir, "extraction", filename[0:-4] + "_bathy_prov.laz"), in_data_bathy, extraField=extra)

    if len(in_data_topo) > 1:
        pl.lastools.WriteLAS(os.path.join(odir, "extraction", filename[0:-4] + "_topo_prov.laz"), in_data_topo)


workspace = r'G:\RENNES1\Loire_totale_automne2019\Loire_Sully-sur-Loire_Checy\05-Traitements\C3'
list_c3 = glob.glob(os.path.join(workspace, "*_C3_r.laz"))
water_surface = r'G:\RENNES1\Loire_totale_automne2019\Loire_Sully-sur-Loire_Checy\05-Traitements\C2_ground_thin_1m_watersurface_smooth5.laz'

params_CC = ['standard','LAS','Loire45-3']

query = pl.cloudcompare.c2c_files(params_CC, list_c3, water_surface, 10, 3)

list_c2c = glob.glob(os.path.join(workspace, "*C3_r_C2C.laz"))
print("%i files found !" %len(list_c2c))
Parallel(n_jobs=10,verbose=2)(delayed(extract_bathy)(file, [0.25,-5], 100) for file in list_c2c)


