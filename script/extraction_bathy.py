# coding: utf-8
# Baptiste Feldmann
import numpy as np
import plateforme_lidar as pl
import glob
import os
from joblib import Parallel, delayed

def extract_bathy(workspace,filename,filter_V=[0.25,-10],filter_H=250):
    inData=pl.lastools.readLAS_laspy(workspace+filename,extraField=True)
    dist_plani=np.sqrt((inData['c2c_absolute_distances_x']**2)+(inData['c2c_absolute_distances_y']**2))
    
    select=np.logical_and(np.logical_and(inData['c2c_absolute_distances_z']<filter_V[0],
                                         inData['c2c_absolute_distances_z']>filter_V[1]),
                                         dist_plani<filter_H)
    inData_bathy=pl.lastools.Filter_LAS(inData,select)
    inData_topo=pl.lastools.Filter_LAS(inData,np.logical_not(select))
    
    extra=[(("depth","float32"),np.round(inData_bathy['c2c_absolute_distances_z'],decimals=2))]
    if len(inData_bathy)>1:
        pl.lastools.writeLAS(workspace+"extraction/"+filename[0:-4]+"_bathy_prov.laz",inData_bathy,extraField=extra)

    if len(inData_topo)>1:
        pl.lastools.writeLAS(workspace+"extraction/"+filename[0:-4]+"_topo_prov.laz",inData_topo)


workspace=r'G:\RENNES1\Loire_totale_automne2019\Loire_Sully-sur-Loire_Checy\05-Traitements\C3'+'//'
liste=glob.glob(workspace+"*_C3_r.laz")
liste_noms=[os.path.split(f)[1] for f in liste]
surface_eau=r'G:\RENNES1\Loire_totale_automne2019\Loire_Sully-sur-Loire_Checy\05-Traitements\C2_ground_thin_1m_watersurface_smooth5.laz'

params_CC=['standard','LAS','Loire45-3']

query=pl.cloudcompare.c2c_files(params_CC,workspace,liste_noms,surface_eau,10,3)

liste=glob.glob(workspace+"*C3_r_C2C.laz")
liste_noms=[os.path.split(f)[1] for f in liste]
print("%i files found !" %len(liste_noms))

Parallel(n_jobs=10,verbose=2)(delayed(extract_bathy)(workspace,i,[0.25,-5],100) for i in liste_noms)


    
    



        
    


        
