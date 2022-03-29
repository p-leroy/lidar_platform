# coding: utf-8
# Baptiste Feldmann
# Module pour le traitement du fichier SBET
from . import utils
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import pyproj
import math,struct,os,mmap,copy

# lat : 41.9875 - 51.5125
# long : -5.5 - 8.5
# nrows= 381 ; ncols= 421
##borne_lat=[41.9875,51.5125]
##borne_long=[-5.5167,8.5167]
##pts0=[-5.5167,51.5125]
##
##delta_lat=0.025
##delta_long=1/30

def Merge_sbet(listSBET):
    new=copy.deepcopy(listSBET[0])
    for i in range(1,len(listSBET)):
        new.array=np.append(new.array,listSBET[i].array)

    new.gps_time=new.array['time']
    new.latitude=new.array['latitude']
    new.longitude=new.array['longitude']
    new.elevation=new.array['elevation']
    return new
    
class SBET(object):
    def __init__(self,filepath):
        self.filepath=filepath

        if os.path.splitext(self.filepath)!=".out":
            self.open_file()
        else:
            raise OSError("Unknown file extension, can read only .out file !")
    
    def __str__(self):
        return "\n".join(self.__dict__.keys())
    
    def open_file(self):
        f=open(self.filepath,mode='rb')
        f_size=os.path.getsize(self.filepath)
        data=mmap.mmap(f.fileno(),f_size,access=mmap.ACCESS_READ)
        nbr_line=int(len(data)/utils.LINE_SIZE)

        temp=[]
        for i in range(0,nbr_line):
            temp+=[struct.unpack('17d',data[i*utils.LINE_SIZE:(i+1)*utils.LINE_SIZE])]
        self.array=np.array(temp,dtype=utils.LIST_OF_ATTR)
        self.array['latitude']*=180/math.pi
        self.array['longitude']*=180/math.pi

        self.gps_time=self.array['time']
        self.latitude=self.array['latitude']
        self.longitude=self.array['longitude']
        self.elevation=self.array['elevation']
    
    def _compute_undulation(self,geoidgrid):
        # npzfile=np.load("G:/RENNES1/BaptisteFeldmann/Vertical_datum/"+geoidgrid+".npz")
        npzfile=np.load(utils.VERTICAL_DATUM_DIR+geoidgrid+".npz")
        grille=npzfile[npzfile.files[0]]
        undulation=griddata(grille[:,0:2],grille[:,2],(self.longitude,self.latitude),method='linear')
        return undulation

    def h2He(self,geoidgrid):
        # ellipsoidal height to altitude
        undulation=self._compute_undulation(geoidgrid)
        alti=self.elevation-undulation
        self.elevation,self.array['elevation']=alti,alti
    
    def He2h(self,geoidgrid):
        # altitude to ellipsoidal height
        undulation=self._compute_undulation(geoidgrid)
        height=self.elevation+undulation
        self.elevation,self.array['elevation']=height,height

    def projection(self,epsg_IN,epsg_OUT):
        #----Conversion coordonnées Géo vers Projetées----------#
        # epsg:4171 -> ETRS89-géo lat,long,h
        # epsg:2154 -> RGF93-L93 E,N,H
        # epsg:4326 -> WGS84-géo lat,long,h
        # epsg:32634 -> WGS84-UTM 34N E,N,H
        # epsg:4167 -> NZGD2000 lat,long,h
        # epsg:2193 -> NZTM2000 E,N,H
        transformer=pyproj.Transformer.from_crs(epsg_IN,epsg_OUT)
        self.easting,self.northing=transformer.transform(self.latitude,self.longitude)
    
    def export(self,epsg_in,epsg_out):
        if not hasattr(self,"easting"):
            self.projection(epsg_in,epsg_out)

        data=np.array([self.easting,self.northing,self.elevation,self.gps_time])
        f=np.savetxt(self.filepath[0:-4]+"_ascii.txt",np.transpose(data),fmt="%.3f;%.3f;%.3f;%f",delimiter=";",header="X;Y;Z;gpstime")

    def interpolate(self,time_ref):
        temp=np.transpose([self.easting,self.northing,self.elevation])
        f=interp1d(self.gps_time,temp,axis=0)
        return f(time_ref)

def calc_grid(name_geoid,pts0,deltas):
    """
    Function for compute a geoid grid from an Ascii file.
    To use a geoid grid, you have to register a NPZ file
    with a specific format. To formalize the geoid grid file,
    use this function.

    Parameters
    ----------
    name_geoid : str
            name of the Ascii file that you want to formalize
            ex : "RAF09"
    pts0 : ndarray
           geographical coordinates of the top left point
    deltas : ndarray
           steps sampling between grid points

    Return
    ------
    Register a formalize geoid grid in NPZ file
    """
    npzfile=np.load("D:/TRAVAIL/Vertical_datum/"+name_geoid+"_brut.npz")
    grille=npzfile[npzfile.files[0]]
    taille=np.shape(grille)
    tableau=np.zeros((taille[0]*taille[1],3))
    compteur=0
    for lig in range(0,taille[0]):
        for col in range(0,taille[1]):
            tableau[compteur,0]=round(pts0[0]+deltas[0]*col,6)
            tableau[compteur,1]=round(pts0[1]-deltas[1]*lig,6)
            tableau[compteur,2]=grille[lig,col]
            compteur+=1
    np.savez_compressed("D:/TRAVAIL/Vertical_datum/RAF09.npz",tableau)
    return True

def Projection(epsg_in,epsg_out,x,y,z):
    """
    Function for compute the transformation between
    2 references system. This function use PyProj library.

    Parameters
    ----------
    epsg_in : int
              EPSG code of source reference system
              ex : 4171
    epsg_out : int
               EPSG code of target reference system
               ex : 2154
    x : ndarray
        Abscissa axis data.
        For geographical coordinates it's latitude
        For projection coordinates it's Easting
    y : ndarray
        Ordinate axis data.
        For geographical coordinates it's longitude
        For projection coordinates it's Northing
    z : ndarray
        Elevation data
        This field is required but not modified
        
    Return
    ------
    result : ndarray
            table of the transform data
    """
    transformer=pyproj.Transformer.from_crs(epsg_in,epsg_out)
    temp=transformer.transform(x,y,z)
    result=np.array([temp[0],temp[1],temp[2]])
    return np.transpose(result)

def Sbet_config(filepath):
    sbet_dict={}
    for i in np.loadtxt(filepath,str,delimiter="="):
        sbet_dict[i[0]]=i[1]
    
    listObj=[]
    for i in sbet_dict['listFiles'].split(','):
        listObj+=[SBET(sbet_dict['path']+str(i))]
    
    if len(listObj)>1:
        sbet_obj=Merge_sbet(listObj)
    else:
        sbet_obj=listObj[0]
    
    if sbet_dict['Z']=='height':
        sbet_obj.h2He(sbet_dict['geoidgrid'])
    sbet_obj.projection(int(sbet_dict['epsg_source']),int(sbet_dict['epsg_target']))
    return sbet_obj
