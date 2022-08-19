# coding: utf-8
# Baptiste Feldmann
import numpy as np
import lidar_traitements as LT
import glob
import os
from joblib import Parallel, delayed

def extract_bathy(workspace,filename,filter_V=[0.25,-10],filter_H=250):
    data,metadata=LT.lastools.read(workspace + filename, extra_field=True)
    names=metadata['col_names']
    select=np.logical_and(data[:,names.index('c2c_absolute_distances_(z)')]<filter_V[0],
                          data[:,names.index('c2c_absolute_distances_(z)')]>filter_V[1])
    temp=data[select,:]
    autre=data[np.logical_not(select),:]
    
    dist_plani=np.sqrt((temp[:,names.index('c2c_absolute_distances_(x)')]**2)+(temp[:,names.index('c2c_absolute_distances_(y)')]**2))
    select=dist_plani<filter_H
    data_bathy=temp[select,:]
    data_topo=np.vstack([autre,temp[np.logical_not(select),:]])
    
    extra=[(("depth","float32"),np.round(data_bathy[:,names.index('c2c_absolute_distances_(z)')],decimals=2))]
    if len(data_bathy[:,0])>1:
        #LT.lastools.writeLAS(workspace+"extraction/"+filename[0:-4]+"_in.laz",data_bathy,vlrs=metadata['vlrs'],extra_field=extra)
        LT.lastools.WriteLAS(workspace + "extraction/" + filename[0:-4] + "_in.laz", data_bathy, vlrs=metadata['vlrs'])

##    if len(data_topo[:,0])>1:
##        LT.lastools.writeLAS(workspace+"extraction/"+filename[0:-4]+"_topo_prov.laz",data_topo,vlrs=metadata['vlrs'])


workspace="G:/RENNES1/BaptisteFeldmann/Python/training/Loire/2020_zone49-1/classif_C3/ground/"
liste=glob.glob(workspace+"*.laz")
liste_noms=[os.path.split(f)[1] for f in liste]
surface_eau="G:/RENNES1/BaptisteFeldmann/Python/training/Loire/2020_zone49-1/surface_water_C2.laz"

params_CC=['standard','LAS','Loire49-1']

query=LT.cloudcompare.c2c_files(params_CC,workspace,liste_noms,surface_eau,10,5)

liste=glob.glob(workspace+"*_C2C.laz")
liste_noms=[os.path.split(f)[1] for f in liste]
print("%i files found !" %len(liste_noms))

Parallel(n_jobs=10,verbose=1)(delayed(extract_bathy)(workspace,i,[0.25,-0.5],100) for i in liste_noms)


    
    



        
    


        
