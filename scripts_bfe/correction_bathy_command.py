# coding: utf-8
# Baptiste Feldmann
import plateforme_lidar as PL
import numpy as np
import glob,os,time,shutil
from joblib import Parallel,delayed

def corbathy_discret(filepath,data_sbet):
    offsetName=-10
    output_suffix="_corbathy"
    # Ouverture du fichier contenant la bathy
    inData= tools.lastools.read(filepath, extra_field=True)

    select=inData.depth<0.01
    dataUnderWater= tools.lastools.filter_las(inData, select)
    dataAboveWater= tools.lastools.filter_las(inData, np.logical_not(select))
    del inData
    
    gpsTime=dataUnderWater.gps_time
    depthApp=dataUnderWater.depth
    
    # Calcul des nouvelles positions
    dataInterp= topo_bathymetry.sbet.interpolate(data_sbet[0], data_sbet[1], gpsTime)
    coordsTrue,depthTrue=PL.calculs.correction_3d(dataUnderWater.XYZ, depthApp, dataInterp[:, 0:3])
    
    # Ecriture des résultats dans les fichiers LAS
    depthAll=np.concatenate((np.round(depthTrue,decimals=2),np.array([None]*len(dataAboveWater))))
    extra=[(("depth","float32"),depthAll)]
    dataUnderWater.XYZ=coordsTrue
    data_corbathy= tools.lastools.merge_las([dataUnderWater, dataAboveWater])
    tools.lastools.WriteLAS(filepath[0:offsetName] + output_suffix + ".laz", data_corbathy, format_id=1, extraField=extra)

def corbathy_fwf(filepath):
    offsetName=-10
    output_suffix="_corbathy"
    # Ouverture du fichier contenant la bathy
    inData= tools.lastools.readLAS_laspy(filepath, True)

    vectApp=np.vstack([inData.x_t,inData.y_t,inData.z_t]).transpose()
    vectTrue_all=PL.calculs.correction_vect(vectApp)
    inData.x_t,inData.y_t,inData.z_t=vectTrue_all[:,0],vectTrue_all[:,1],vectTrue_all[:,2]
    
    select=inData.depth<0.01
    dataUnderWater= tools.lastools.filter_las(inData, select)
    dataAboveWater= tools.lastools.filter_las(inData, np.logical_not(select))
    vectAppUnderWater=vectApp[select]
    del inData

    depthApp=dataUnderWater.depth
    # Calcul des nouvelles positions
    coordsTrue,depthTrue=PL.calculs.correction_3d(dataUnderWater.XYZ, depthApp, vectorApp=vectAppUnderWater)
    
    # Ecriture des résultats dans les fichiers LAS
    depthAll=np.concatenate((np.round(depthTrue,decimals=2),np.array([None]*len(dataAboveWater))))
    extra=[(("depth","float32"),depthAll)]

    dataUnderWater.XYZ=coordsTrue
    dataCorbathy= tools.lastools.merge_las([dataUnderWater, dataAboveWater])

    #return data_corbathy,extra,metadata['vlrs']
    #PL.lastools.writeLAS(filepath[0:offsetName]+"_corbathy2.laz",dataCorbathy,format_id=4,extraField=extra)
    tools.lastools.WriteLAS(filepath[0:offsetName] + output_suffix + ".laz", dataCorbathy, format_id=9)
    shutil.copyfile(filepath[0:-4]+".wdp",filepath[0:offsetName]+output_suffix+".wdp")

#============================================================#
#----Etape 0: Preparation des parametres du projet-----------#
#============================================================#
##import argparse
##parser=argparse.ArgumentParser(description='Process some strings...')
##
##parser.add_argument('-i', metavar='N', type=str)
##parser.add_argument('-sbet', metavar='N', type=str)
##parser.add_argument('-fwf',action='store_true')
##parser.add_argument('-n_jobs', metavar='N', type=int,default=1)
##
##args=parser.parse_args()
##chemin=args.i
##opt_fwf=args.fwf
##file_sbet=args.sbet
##cores=args.n_jobs

chemin=r'G:\RENNES1\Aude_fevrier2020\05-Traitements\C3\classification\vol_bathy\dalles_nocor\transfert\Aude_C3_class16_nocor.laz'
file_sbet="params_sbet.txt"
opt_fwf=False
cores=1

workspace=os.path.split(chemin)[0]+"/"

list_path=glob.glob(chemin)
list_las_files=[os.path.split(i)[1] for i in list_path]
print("[Bathymetric correction] : "+str(len(list_path))+" files found !")
debut=time.time()

if opt_fwf:
    print("[Bathymetric correction] : FWF mode Waiting...")
    if len(list_las_files)==1:
        corbathy_fwf(workspace+list_las_files[0])
    else:
        Parallel(n_jobs=cores,verbose=1)(delayed(corbathy_fwf)(workspace+f) for f in list_las_files)
else:
    print("[Bathymetric correction] : SBET data processes, waiting...",end='\r')
    sbetTime,sbetCoords= topo_bathymetry.sbet.sbet_config(workspace + file_sbet)
    print("[Bathymetric correction] : SBET data processes, done !")
    print("[Bathymetric correction] : Discrete mode Waiting...")
    if len(list_las_files)==1:
        corbathy_discret(workspace+list_las_files[0],[sbetTime,sbetCoords])
    else:
        Parallel(n_jobs=cores,verbose=1)(delayed(corbathy_discret)(workspace+f,[sbetTime,sbetCoords]) for f in list_las_files)
 
fin=time.time()
print("[Bathymetric correction] : Complete in "+str(round(fin-debut,1))+" sec")
