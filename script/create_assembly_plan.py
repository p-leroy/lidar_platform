# coding: utf-8
# Baptiste Feldmann
import simplekml
import glob
import os
import pyproj
import numpy as np

workspace=r'G:\RENNES1\Loire_octobre2020_Rtemus\06-Livrables\PointClouds\C3'+'//'

liste_files=glob.glob(workspace+"*.laz")
scale=1000
coords_loc=4
date="19/10/2020"
rootname="Loire_19102020_RTemus"

liste_names=[]
for i in liste_files :
    liste_names+=[os.path.split(i)[1][0:-4]]


plan=simplekml.Kml()

for nom in liste_names:
    coord=np.int_(nom.split(sep="_")[coords_loc:coords_loc+2])
    coords_line=np.array([[coord[0],coord[1]],
                          [coord[0],coord[1]+scale],
                          [coord[0]+scale,coord[1]+scale],
                          [coord[0]+scale,coord[1]],
                          [coord[0],coord[1]]])
    p1=pyproj.Proj(init="epsg:2154")
    p2=pyproj.Proj(init="epsg:4171")
    tmp=pyproj.transform(p1,p2,coords_line[:,0],coords_line[:,1])
    
    pol=plan.newpolygon(name="dalle_"+"_".join(nom.split("_")[coords_loc:coords_loc+2]))
    pol.outerboundaryis=[(tmp[0][0],tmp[1][0]),
                         (tmp[0][1],tmp[1][1]),
                         (tmp[0][2],tmp[1][2]),
                         (tmp[0][3],tmp[1][3]),
                         (tmp[0][0],tmp[1][0])]

    pol.style.linestyle.color=simplekml.Color.red
    pol.style.linestyle.scale=2
    pol.style.linestyle.width=3
    pol.description="Date : "+date+"\nCRS : RGF93-Lambert93\nVertical datum : NGF-IGN69\n"
    pol.style.polystyle.color=simplekml.Color.hexa('00000000')

    
plan.save(workspace+"tableau_assemblage_"+rootname+".kml")
