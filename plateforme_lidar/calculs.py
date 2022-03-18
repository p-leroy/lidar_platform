# coding: utf-8
# Baptiste Feldmann
# Liste de fonctions pour la correction bathy
from . import lastools,utils
import pylas

import numpy as np
import glob,os,simplekml,itertools
import matplotlib.pyplot as plt
import scipy.spatial as scp
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import shapely
import shapely.ops
from shapely.geometry import Polygon
from joblib import Parallel,delayed

def correction3D(pt_app,depthApp,pt_shot=[],vectorApp=[],indRefr=1.333):
    """Bathymetric correction 3D

    Args:
        pt_app (numpy.ndarray): apparent points
        depthApp (numpy.ndarray): apparent depth
        pt_shot (list, optional): coordinates for each laser shot, useful in discrete mode. Defaults to [].
        vectorApp (list, optional): apparent vector shot, useful in fwf mode. Defaults to [].
        indRefr (float, optional): water refraction indice. Defaults to 1.333.

    Raises:
        ValueError: pt_shot and vectorApp shouldn't be Null both

    Returns:
        true point coordinates (numpy.ndarray)
        true depth (numpy.ndarray)
    """
    if len(pt_shot)>0 and len(vectorApp)==0:
        #discret mode
        vectApp=pt_shot-pt_app
    elif len(pt_shot)==0 and len(vectorApp)>0:
        #fwf mode
        vectApp=np.copy(vectorApp)
    else:
        raise ValueError("pt_shot and vectorApp shouldn't be Null both")
    vectApp_norm=np.linalg.norm(vectApp,axis=1)

    # compute "gisement" with formula that removes ambiguity of pi radians on the calculation of 'arctan'
    gisement_vect=2*np.arctan(vectApp[:,0]/(np.linalg.norm(vectApp[:,0:2],axis=1)+vectApp[:,1]))
    thetaApp=np.arccos(vectApp[:,2]/vectApp_norm)
    thetaTrue=np.arcsin(np.sin(thetaApp)/indRefr)
    depthTrue=depthApp*np.cos(thetaApp)/(indRefr*np.cos(thetaTrue))

    distPlan=depthApp*np.tan(thetaApp)-depthTrue*np.tan(thetaTrue)
    coords=np.vstack([pt_app[:,0]+distPlan*np.sin(gisement_vect),
                      pt_app[:,1]+distPlan*np.cos(gisement_vect),
                      pt_app[:,2]+depthTrue-depthApp])

    return np.transpose(coords),depthTrue

def correction_vect(vectorApp,indRefr=1.333):
    """bathymetric correction only for vector shot (in fwf mode)

    Args:
        vectorApp (numpy.ndarray): apparent vector shot, useful in fwf mode
        indRefr (float, optional): water refraction indice. Defaults to 1.333.

    Returns:
        true vector shot (numpy.ndarray)
    """
    # bathymetric laser shot correction for fwf lidar data 
    vectApp_norm=np.linalg.norm(vectorApp,axis=1)
    vectTrue_norm=vectApp_norm/indRefr

    # compute "gisement" with formula that removes ambiguity of pi radians on the calculation of 'arctan'
    gisement_vect=2*np.arctan(vectorApp[:,0]/(np.linalg.norm(vectorApp[:,0:2],axis=1)+vectorApp[:,1]))
    thetaApp=np.arccos(vectorApp[:,2]/vectApp_norm)
    thetaTrue=np.arcsin(np.sin(thetaApp)/indRefr)
    vectTrue=np.vstack([vectTrue_norm*np.sin(thetaTrue)*np.sin(gisement_vect),
                         vectTrue_norm*np.sin(thetaTrue)*np.cos(gisement_vect),
                         vectTrue_norm*np.cos(thetaTrue)])
    return np.transpose(vectTrue)

#======================================#

class Alphashape(object):
    def __init__(self,points2D,alpha=100):
        self.points=points2D
        self.alpha=alpha
        self.compute_alphashape()
        
    def compute_alphashape(self):
        # More you decrease alpha factor, greater the constraint will be on alphashape
        # More you increase alpha factor, more the alphashape will be a convex hull
        print("[Alphashape] Delaunay triangulation...",end='')
        tri=scp.Delaunay(self.points)
        nbrVertices=len(tri.vertices)
        print("done !")

        print("[Alphashape] Cleaning %d triangles :" %nbrVertices)
        displayer=utils.Timing(nbrVertices,20)
        
        edges=set()
        edge_points=[]
        for i in range(0,nbrVertices):
            msg=displayer.timer(i)
            if len(msg)>1:
                print("[Alphashape] "+msg)
                
            idx=tri.vertices[i]
            triangle=self.points[idx,:]
            length=[np.linalg.norm(triangle[i%3]-triangle[(i+1)%3]) for i in [0,1,2]]
            s=np.sum(length)*0.5
            area=s*np.prod(s-length)
            if area>0:
                area=np.sqrt(area)

            #test if circumradius is lower than alpha parameter
            if np.prod(length)/(4*area) < self.alpha:
                for i,j in itertools.combinations(idx,r=2):
                    if (i,j) not in edges and (j,i) not in edges:
                        edges.add((i,j))
                        edge_points.append(self.points[[i,j],:])
        print("[Alphashape] Cleaning %d triangles : done !" %nbrVertices)

        print("[Alphashape] Polygonize...",end='')
        m=shapely.geometry.MultiLineString(edge_points)
        triangles=list(shapely.ops.polygonize(m))
        self.alphashape=shapely.ops.cascaded_union(triangles)
        print("done !")

    def viewer(self):
        if type(self.alphashape)==shapely.geometry.polygon.Polygon:
            a=self.alphashape.exterior.xy
            plt.fill(a[0],a[1],alpha=2,edgecolor="red",facecolor="blue")
        else:
            for i in self.alphashape:
                a=i.exterior.xy
                plt.fill(a[0],a[1],alpha=2,edgecolor="red",facecolor="blue")
        plt.show()

def computeDBSCAN(filepath,maxdist=1,minsamples=5):
    """make Scikit-Learn DBSCAN clustering
    (see docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

    Args:
        filepath (str): path to LAS file
        mindist (int, optional): Maximum distance between two samples. Defaults to 1.
        minsamples (int, optional): Minimum number of samples in each cluster. Defaults to 5.
    """
    data=lastools.readLAS(filepath)
    model=DBSCAN(eps=maxdist,min_samples=minsamples,algorithm='kd_tree',leaf_size=1000,n_jobs=46).fit(data.XYZ)
    
    if len(np.unique(model.labels_))>1:
        extra=[(("labels","int16"),model.labels_)]
        print("Number of clusters : "+str(len(np.unique(model.labels_))-1))
        lastools.writeLAS(filepath[0:-4]+"_DBSCAN.laz",data,extraField=extra)
    else:
        print("DBSCAN find only 1 cluster !")

def computeDensity(points,core_points=[],radius=1,p_norm=2):
    """counting points in neighborhood
    With scipy.spatial.cKDTree

    Args:
        data (numpy.ndarray): input coordinates
        core_points (numpy_ndarray): points for which density will be calculted.
                                    If core_points is empty density will be calculted for all points.
        radius (float): neighbor searching radius
        p (integer): order of the norm for Minkowski distance. Default= 2

    Returns:
        density (integer): number of points
    """
    tree=cKDTree(points,leafsize=1000)
    if len(core_points)==0:
        core_points=np.copy(points)

    return tree.query_ball_point(core_points,r=radius,p=p_norm,return_length=True)
    
def merge_c2c_fwf(workspace,fichier):
    tab_fwf,metadata_fwf=lastools.readLAS(workspace+fichier,"fwf")
    tab_extra,metadata_extra=lastools.readLAS(workspace+fichier[0:-4]+"_extra.laz","standard",True)
    names_fwf=metadata_fwf['col_names']
    names_extra=metadata_extra['col_names']
    
    controle=np.sqrt((tab_fwf[:,0]-tab_extra[:,0])**2+(tab_fwf[:,1]-tab_extra[:,1])**2+(tab_fwf[:,2]-tab_extra[:,2])**2)
    try: assert(all(controle<0.003))
    except:
        raise ValueError("LAS_FWF file and LAS file don't match exactly!\nPlease check your files...")

    dist_Z=tab_extra[:,names_extra.index('c2c_absolute_distances_(z)')]
    dist_plani=np.sqrt(np.power(tab_extra[:,names_extra.index('c2c_absolute_distances_(x)')],2)+np.power(tab_extra[:,names_extra.index('c2c_absolute_distances_(y)')],2))
    num=names_fwf.index("wave_packet_desc_index")
    tab_tot=np.hstack([tab_extra[:,0:-4],tab_fwf[:,num::],np.reshape(dist_Z,(len(dist_Z),1)),np.reshape(dist_plani,(len(dist_plani),1))])
    names_tot=names_extra[0:-4]+names_fwf[num::]+['depth','distance_H']
    return tab_tot,names_tot,metadata_fwf['vlrs']
  
def select_pairs_overlap(filepath,shifts):
    liste_files=glob.glob(filepath)
    liste_polygon=[]
    liste_num=[]
    for i in liste_files:
        data=lastools.readLAS(i)
        liste_num+=[str(os.path.split(i)[1][shifts[0]:shifts[0]+shifts[1]])]
        pca_pts=PCA(n_components=2,svd_solver='full')
        dat_new=pca_pts.fit_transform(data.XYZ[:,0:2])
        del data
        
        borne=np.array([[min(dat_new[:,0]),min(dat_new[:,1])],
                        [min(dat_new[:,0]),max(dat_new[:,1])],
                        [max(dat_new[:,0]),max(dat_new[:,1])],
                        [max(dat_new[:,0]),min(dat_new[:,1])]])
        borne_new=pca_pts.inverse_transform(borne)
        liste_polygon+=[Polygon(borne_new)]

    comparison={}
    for i in range(0,len(liste_polygon)-1):
        listing=[]
        for c in range(i+1,len(liste_polygon)):
            if liste_polygon[i].overlaps(liste_polygon[c]):
                diff=liste_polygon[i].difference(liste_polygon[c])
                if diff.area/liste_polygon[i].area<0.9:
                    listing+=[liste_num[c]]

        if len(listing)>0:
            comparison[liste_num[i]]=listing

    return comparison

def writeKML(filepath,names,descriptions,coordinates):
    try: assert(len(names)==len(descriptions) and len(names)==len(coordinates) and len(descriptions)==len(coordinates))
    except : print("Taille différente pour names, description et coords !!")
    fichier=simplekml.Kml()
    for i in names:
        fichier.newpoint(name=i,description=descriptions[names.index(i)],coords=[coordinates[names.index(i)]])
    fichier.save(filepath)
    return True

class ReverseTiling_mem(object):
    def __init__(self,workspace,fileroot,buffer=False,cores=50,write_ptSrcId=True):
        print("[Reverse Tiling memory friendly]")
        self.workspace=workspace
        self.cores=cores
        self.motif=fileroot.split(sep='XX')
        self.write_ptSrcId=write_ptSrcId
        
        print("Remove buffer...",end=" ")
        if buffer:
            self.removeBuffer()
        print("done !")

        print("Searching flightlines...",end=" ")
        self.searchingLines()
        print("done !")

        print("Writing flightlines...",end="")
        self.writingLines(buffer)
        print("done !")  

    def _get_ptSrcId(self,filename):
        f=pylas.read(self.workspace+filename)
        pts_srcid=np.unique(f.pt_src_id)
        return [filename,pts_srcid]

    def _mergeLines(self,key,maxLen,linenum=0):
        query="lasmerge -i "
        for filename in self.linesDict[key]:
            query+=self.workspace+filename+" "

        if linenum==0:
            diff=maxLen-len(str(key))
            numLine=str(key)[0:-2]
        else:
            numLine=str(linenum)
            diff=maxLen-len(numLine)

        name=self.motif[0]+"0"*diff+numLine+self.motif[1]
        query+="-keep_point_source "+str(key)+" -o "+self.workspace+name
        utils.Run(query)

    def removeBuffer(self):
        os.mkdir(self.workspace+"new_tile")
        query="lastile -i "+self.workspace+"*.laz -remove_buffer -cores "+str(self.cores)+" -odir "+self.workspace+"new_tile -olaz"
        utils.Run(query)
        self.workspace+="new_tile/"

    def searchingLines(self):
        listNames=[os.path.split(i)[1] for i in glob.glob(self.workspace+"*.laz")]
        result=Parallel(n_jobs=self.cores,verbose=0)(delayed(self._get_ptSrcId)(i) for i in listNames)
        self.linesDict={}
        for i in result:
            for c in i[1]:
                if c not in self.linesDict.keys():
                    self.linesDict[c]=[i[0]]
                else:
                    self.linesDict[c]+=[i[0]]

    def writingLines(self,buffer):
        maxPtSrcId=len(str(max(self.linesDict.keys())))
        maxNumberLines=len(str(len(self.linesDict.keys())+1))
        if self.write_ptSrcId:
            for i in self.linesDict.keys():
                print(i)
                self._mergeLines(i,maxPtSrcId)
        else:
            num=1
            list_ptsrcid=list(self.linesDict.keys())
            list_ptsrcid.sort()
            for i in list_ptsrcid:
                print(i)
                self._mergeLines(i,maxNumberLines,num)
                num+=1

        if buffer:
            listNames=[os.path.split(i)[1] for i in glob.glob(self.workspace+self.motif[0])]
            for filename in listNames:
                os.rename(self.workspace+filename,self.workspace[0:-9]+filename)

            for filepath in glob.glob(self.workspace+"*_1.laz"):
                os.remove(filepath)
            os.rmdir(self.workspace)

'''class ReverseTiling_fast(object):
    def __init__(self,workspace,fileroot,buffer=False,cores=50):
        print("[Reverse Tiling fastest version]")
        self.workspace=workspace
        self.cores=cores
        self.motif=fileroot.split(sep='XX')

        print("Remove buffer...",end=" ")
        if buffer:
            self.removeBuffer()
        print("done !")

        print("Searching flightlines...",end=" ")
        self.searchingLines()
        print("done !")

        print("Writing flightlines...",end="")
        self.writingLines(buffer)
        print("done !") 

    def removeBuffer(self):
        query="lastile -i "+self.workspace+"*.laz -remove_buffer -cores "+str(self.cores)+" -olaz"
        utils.Run(query)
        for filepath in glob.glob(workspace+"*_1.laz"):
            nom=os.path.split(filepath)[-1]
            os.rename(workspace+nom,workspace+"new_tile/"+nom)
        workspace+="new_tile/"

    def searchingLines(self):
        self.linesDict={}
        listData=[lastools.readLAS(i) for i in glob.glob(workspace+"*.laz")]'''

