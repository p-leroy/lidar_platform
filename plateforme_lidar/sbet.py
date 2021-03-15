# coding: utf-8
# Baptiste Feldmann
# Module pour le traitement du fichier SBET
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import pyproj
import math,struct,os,mmap

# lat : 41.9875 - 51.5125
# long : -5.5 - 8.5
# nrows= 381 ; ncols= 421
##borne_lat=[41.9875,51.5125]
##borne_long=[-5.5167,8.5167]
##pts0=[-5.5167,51.5125]
##
##delta_lat=0.025
##delta_long=1/30

def h2He(geoidgrid,long,lat,hauteur):
    """
    Function for convert ellipsoidal height into altitude
    This function uses linear interpolation between points
    in geoidgrid and your points

    Parameters
    ----------
    geoidgrid : str
            filename (.npz) of the geoid grid that you want to use
            ex : "RAF09.npz"
    long : ndarray
           longitudes of points
    lat : ndarray
          latitudes of points
    hauteur : ndarray
              ellispoidal height of points

    Return
    ------
    alti : ndarray
        The altitudes of points
    """
    npzfile=np.load("G:/RENNES1/BaptisteFeldmann/Vertical_datum/"+geoidgrid+".npz")
    grille=npzfile[npzfile.files[0]]
    ondulation=griddata(grille[:,0:2],grille[:,2],(long,lat),method='linear')
    alti=hauteur-ondulation
    return alti

def He2h(geoidgrid,long,lat,altitude):
    """
    Function for convert altitude into ellipsoidal height
    This function uses linear interpolation between points
    in geoidgrid and your points

    Parameters
    ----------
    geoidgrid : str
            filename (.npz) of the geoid grid that you want to use
            ex : "RAF09.npz"
    long : ndarray
           longitudes of points
    lat : ndarray
          latitudes of points
    altitude : ndarray
               altitude of points

    Return
    ------
    hauteur : ndarray
        The ellipsoidal heights of points
    """
    npzfile=np.load("G:/RENNES1/BaptisteFeldmann/Vertical_datum/"+geoidgrid+".npz")
    grille=npzfile[npzfile.files[0]]
    ondulation=griddata(grille[:,0:2],grille[:,2],(long,lat),method='linear')
    hauteur=altitude+ondulation
    return hauteur

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

def open_sbet(filepath,register=False):
    """
    Function for open a binary sbet file and return
    data in a matrix.

    Parameters
    ----------
    workspace : str
            path of your work space
    file : str
           name of your binary sbet file
    register : bool, optional
           if you want to register the data in a file

    Returns
    ------
    field_names : ndarray
                table of the field names
    tab : ndarray
         data
    """
    if os.path.splitext(filepath)[1]==".out":
        # 17 attributes of 8 bytes each = 136 bytes
        line_size=int(136)

        f=open(filepath,mode='rb')
        f_size=os.path.getsize(filepath)
        data=mmap.mmap(f.fileno(),f_size,access=mmap.ACCESS_READ)

        try :
            assert(len(data)%line_size==0)
        except :
            raise OSError("Problem with size of file\nIt must contain 17 fields of 8 bytes each !")
        
        nbr_line=int(len(data)/line_size)
        tab=[]
        for i in range(0,nbr_line):
            tab+=[list(struct.unpack('17d',data[i*line_size:(i+1)*line_size]))]

        tab=np.array(tab)
        tab[:,1]=tab[:,1]*180/math.pi
        tab[:,2]=tab[:,2]*180/math.pi

    elif os.path.splitext(filepath)[1]==".npz":
        f=np.load(filepath)
        tab=f[f.files[0]]
    else:
        raise OSError("Unknown file extension, can read only .out or .npz file !")
            
    if register :
        np.savez_compressed(filepath[0:-4]+".npz",tab)
    return tab

def Projection(epsg_in,epsg_out,x,y,z):
    """
    Function for compute the transformation between
    2 references system. This function use PyProj library.

    Parameters
    ----------
    epsg_in : str
              EPSG code of source reference system
              ex : "epsg:2154"
    epsg_out : str
               EPSG code of target reference system
               ex : "epsg:4171"
    x : ndarray
        Abscissa axis data.
        For geographical coordinates it's longitude
    y : ndarray
        Ordinate axis data.
        For geographical coordinates it's latitude
    z : ndarray
        Elevation data
        This field is required but not modified
        
    Return
    ------
    result : ndarray
            table of the transform data
    """
    p1=pyproj.Proj(init=epsg_in)
    p2=pyproj.Proj(init=epsg_out)
    tmp=pyproj.transform(p1,p2,x,y,z)
    result=np.array([tmp[0],tmp[1],tmp[2]])
    return np.transpose(result)

def interpolate(time_sbet,sbet_coords,time_ref,method="linear"):
    """
    Function for interpolate sbet coordinates to get
    exact coordinates of laser shots from GPS Time of
    points cloud.

    Parameters
    ----------
    time_sbet : ndarray
            GPS time of sbet file
    sbet_coords : ndarray
            sbet coordinates
    method : str
        interpolation method
    time_ref : ndarray
        GPS time of points cloud
        
    Return
    ------
    interp : ndarray
            sbet coordinates interpolated
    """
    f=[interp1d(time_sbet,sbet_coords[:,0],kind=method),
       interp1d(time_sbet,sbet_coords[:,1],kind=method),
       interp1d(time_sbet,sbet_coords[:,2],kind=method)]
    interp=np.array([f[0](time_ref),f[1](time_ref),f[2](time_ref),time_ref])
    return np.transpose(interp)

def conversionSbet(listFiles,epsg_SRC,epsg_FIN,change_alti=False):
    """
    Function for sbet treatments. This function open a Sbet file, transform
    coordinates between two references system, uses previous functions.

    Parameters
    ----------
    workspace : str
            path of your work space
    sbet_files : str or list
            name of your sbet file or liste of several sbet files
    epsg_SRC : str
            EPSG code of source reference system
    epsg_FIN : str
            EPSG code of target reference system
    change_alti : str, optional
            name of geoid grid if you want to convert elevation
        
    Returns
    ------
    borne : list
            GPS time bounds
    time_sbet : 
            concatenate GPS time of sbet file(s)
    coords : ndarray
            instrument coordinates of each laser shots 
    """
    if type(listFiles)==str:
        listFiles=[listFiles]

    listData=[open_sbet(i) for i in listFiles]
    mergeData=np.concatenate(listData,axis=0)

    if change_alti!=False:
        temp=h2He(change_alti,mergeData[:,2],mergeData[:,1],mergeData[:,3])
        mergeData[:,3]=temp

    #----Conversion coordonnées Géo vers Projetées----------#
    # epsg:4171 -> ETRS89-géo lat,long,h
    # epsg:2154 -> RGF93-L93 E,N,H
    # epsg:4326 -> WGS84-géo lat,long,h
    # epsg:32634 -> WGS84-UTM 34N E,N,H
    # epsg:4167 -> NZGD2000 lat,long,h
    # epsg:2193 -> NZTM2000 E,N,H
    time_tot=mergeData[:,0]
    if epsg_SRC==epsg_FIN:
        coords_tot=mergeData[:,1:4]
    else:
        coords_tot=Projection(epsg_SRC,epsg_FIN,mergeData[:,2],mergeData[:,1],mergeData[:,3])
    #-------------------------------------------------------#
        
    borne=[min(time_tot),max(time_tot)]
    return borne,time_tot,coords_tot

def Sbet2Ascii(filepath,epsg_src,epsg_target):
    data=open_sbet(filepath)
    coords=Projection(epsg_src,epsg_target,data[:,2],data[:,1],data[:,3])
    output=np.append(coords,np.reshape(data[:,0],(-1,1)),axis=1)
    f=np.savetxt(filepath[0:-4]+"_ascii.txt",output,fmt="%.3f;%.3f;%.3f;%f",delimiter=";",header="X;Y;Z;gpstime")

def Sbet_config(filepath):
    sbet_dict={}
    for i in np.loadtxt(filepath,str,delimiter="="):
        sbet_dict[i[0]]=i[1]

    if sbet_dict['Z']=='height':
        borne,time_sbet,coords_sbet=conversionSbet([sbet_dict['path']+str(i) for i in sbet_dict['listFiles'].split(",")],'epsg:'+sbet_dict['epsg_source'],
                                                   'epsg:'+sbet_dict['epsg_target'],sbet_dict['geoidgrid'])
    else:
        borne,time_sbet,coords_sbet=conversionSbet([sbet_dict['path']+str(i) for i in sbet_dict['listFiles'].split(",")],'epsg:'+sbet_dict['epsg_source'],
                                                   'epsg:'+sbet_dict['epsg_target'])
    return time_sbet,coords_sbet
