# coding: utf-8
# Baptiste Feldmann
import simplekml
import glob
import os
import pyproj
import numpy as np
from sklearn.decomposition.pca import PCA

workspace="G:/RENNES1/New_Zealand_Sept2019/05-Traitements/C1/thin/"

liste=glob.glob(workspace+"*_thin.laz")
liste_noms=[os.path.split(i)[1] for i in liste]
epsg_SRC="epsg:2193"
epsg_CIBLE="epsg:4326"

liste_names=[]
plan=simplekml.Kml()
for i in liste_noms :
    print(i)
    data= tools.lastools.read(workspace + i)
    pca_pts=PCA(n_components=2,svd_solver='full')
    dat_new=pca_pts.fit_transform(data.XYZ[:,0:2])
    borne=np.array([[min(dat_new[:,0]),min(dat_new[:,1])],
                    [min(dat_new[:,0]),max(dat_new[:,1])],
                    [max(dat_new[:,0]),max(dat_new[:,1])],
                    [max(dat_new[:,0]),min(dat_new[:,1])]])
    borne_new=pca_pts.inverse_transform(borne)
    p1=pyproj.Proj(init=epsg_SRC)
    p2=pyproj.Proj(init=epsg_CIBLE)
    tmp=pyproj.transform(p1,p2,borne_new[:,0],borne_new[:,1])
    
    pol=plan.newpolygon(name=i)
    pol.outerboundaryis=[(tmp[0][0],tmp[1][0]),
                         (tmp[0][1],tmp[1][1]),
                         (tmp[0][2],tmp[1][2]),
                         (tmp[0][3],tmp[1][3]),
                         (tmp[0][0],tmp[1][0])]
    pol.style.linestyle.color=simplekml.Color.red
    pol.style.linestyle.scale=2
    pol.style.linestyle.width=3
    pol.description="Système planimétrique : NZTM2000\nSystème altimétrique : NZVD2016"
    pol.style.polystyle.color=simplekml.Color.hexa('0055ff80')

    
plan.save(workspace+"tableau_assemblage_lignes.kml")
