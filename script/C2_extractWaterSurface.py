# coding: utf-8
# Baptiste Feldmann
import plateforme_lidar as PL
import os
import numpy as np
import importlib
importlib.reload(PL)

workspace=r'G:\RENNES1\Loire_totale_automne2019\Loire3_Checy-Saint-Dye\05-Traitements'+'//'
C2_filename="C2_ground_thin_1m.laz"
C3_filename="C3_ground_thin_1m.laz"

CC_opt=['standard','LAS',"Loire45-1"]

# compute rawbathy
filterVert=[-4,-0.3]
query=PL.cloudcompare.open_file(CC_opt,[workspace+C3_filename,workspace+C2_filename])
PL.cloudcompare.c2c_dist(query)
os.remove(PL.cloudcompare.last_file(workspace+C2_filename[0:-4]+"_20*.laz"))
PL.cloudcompare.last_file(workspace+C3_filename[0:-4]+"_C2C_DIST_*.laz",C3_filename[0:-4]+"_C2C.laz")

C3_data=PL.lastools.read(workspace + C3_filename[0:-4] + "_C2C.laz", True)
outData=PL.lastools.filter_las(C3_data, np.logical_and(C3_data.c2c_absolute_distances_z > filterVert[0], C3_data.c2c_absolute_distances_z < filterVert[1]))

density=PL.calculs.computeDensity(outData.XYZ,radius=5)
outData=PL.lastools.filter_las(outData, density > 15)

PL.lastools.WriteLAS(workspace + C3_filename[0:-4] + "_rawbathy.laz", outData)

os.remove(workspace+C3_filename[0:-4]+"_C2C.laz")
del C3_data
del outData

# compute water surface
filterNormal=1
filterDist=25
filterVert=[0.25,3.5]

PL.cloudcompare.compute_normals_dip(workspace+C2_filename,CC_opt,2)
PL.cloudcompare.last_file(workspace+C2_filename[0:-4]+"_20*.laz",C2_filename[0:-4]+"_normals.laz")
query=PL.cloudcompare.open_file(CC_opt,[workspace+C2_filename[0:-4]+"_normals.laz",workspace+C3_filename[0:-4]+"_rawbathy.laz"])
PL.cloudcompare.c2c_dist(query)

os.remove(PL.cloudcompare.last_file(workspace+C3_filename[0:-4]+"_rawbathy_20*.laz"))
PL.cloudcompare.last_file(workspace+C2_filename[0:-4]+"_normals_C2C_DIST_*.laz",C2_filename[0:-4]+"_normals_C2C.laz")

C2_data=PL.lastools.read(workspace + C2_filename[0:-4] + "_normals_C2C.laz", True)

select1=C2_data.dip_degrees<filterNormal
select2=(C2_data.c2c_absolute_distances_x**2+C2_data.c2c_absolute_distances_y**2)**0.5<filterDist
select3=np.logical_and(C2_data.c2c_absolute_distances_z>filterVert[0],C2_data.c2c_absolute_distances_z<filterVert[1])
select_all=np.logical_and(np.logical_and(select1,select2),select3)

outData=PL.lastools.filter_las(C2_data, select_all)
density=PL.calculs.computeDensity(outData.XYZ,radius=5)
outData=PL.lastools.filter_las(outData, density > 15)

PL.lastools.WriteLAS(workspace + C2_filename[0:-4] + "_watersurface.laz", outData)

os.remove(workspace+C2_filename[0:-4]+"_normals.laz")
os.remove(workspace+C2_filename[0:-4]+"_normals_C2C.laz")
