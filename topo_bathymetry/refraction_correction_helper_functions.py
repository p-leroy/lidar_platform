# coding: utf-8
# Paul Leroy
# Baptiste Feldmann
# helper functions for refraction correction

import os
import glob
import itertools

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as scp
from scipy.spatial import cKDTree

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

import shapely
import shapely.ops
from shapely.geometry import Polygon

import simplekml

from ..tools import las, misc


def correction_3d(pt_app, apparent_depth, pt_shot=[], vectorApp=[], indRefr=1.333):
    """Bathymetric correction 3D

    Args:
        pt_app (numpy.ndarray): apparent points
        apparent_depth (numpy.ndarray): apparent depth
        pt_shot (list, optional): coordinates for each laser shot, useful in discrete mode. Defaults to [].
        vectorApp (list, optional): apparent vector shot, useful in fwf mode. Defaults to [].
        indRefr (float, optional): water refraction index. Defaults to 1.333.

    Raises:
        ValueError: pt_shot and vectorApp shouldn't be Null both

    Returns:
        true point coordinates (numpy.ndarray)
        true depth (numpy.ndarray)
    """
    if len(pt_shot) > 0 and len(vectorApp) == 0:  # discrete mode
        print('[calculs.correction3D] discrete mode')
        vect_app = pt_shot - pt_app
    elif len(pt_shot) == 0 and len(vectorApp) > 0:  # FWF mode
        print('[calculs.correction3D] FWF mode')
        vect_app = np.copy(vectorApp)
    else:
        raise ValueError("pt_shot and vectorApp shouldn't be Null both")
    vect_app_norm = np.linalg.norm(vect_app, axis=1)

    # compute "gisement" with formula that removes ambiguity of pi radians on the calculation of 'arctan'
    gisement_vect = 2 * np.arctan(vect_app[:, 0] / (np.linalg.norm(vect_app[:, 0:2], axis=1) + vect_app[:, 1]))
    theta_app = np.arccos(vect_app[:, 2] / vect_app_norm)
    theta_true = np.arcsin(np.sin(theta_app) / indRefr)
    depth_true = apparent_depth * np.cos(theta_app) / (indRefr * np.cos(theta_true))

    dist_plan = apparent_depth * np.tan(theta_app) - depth_true * np.tan(theta_true)
    coords = np.vstack([pt_app[:, 0] + dist_plan * np.sin(gisement_vect),
                        pt_app[:, 1] + dist_plan * np.cos(gisement_vect),
                        pt_app[:, 2] + depth_true - apparent_depth])

    return np.transpose(coords), depth_true


def correction_vect(apparent_vector, refraction_index=1.333):
    """bathymetric correction only for vector shot (in fwf mode)

    Args:
        apparent_vector (numpy.ndarray): apparent vector shot, useful in fwf mode
        refraction_index (float, optional): water refraction index. Defaults to 1.333.

    Returns:
        true vector shot (numpy.ndarray)
    """
    # bathymetric laser shot correction for fwf lidar data
    apparent_vector_norm = np.linalg.norm(apparent_vector, axis=1)
    true_vector_norm = apparent_vector_norm / refraction_index

    # compute "gisement" with formula that removes ambiguity of pi radians on the calculation of 'arctan'
    gisement = 2 * np.arctan(
        apparent_vector[:, 0] / (np.linalg.norm(apparent_vector[:, 0:2], axis=1) + apparent_vector[:, 1])
    )
    apparent_theta = np.arccos(apparent_vector[:, 2] / apparent_vector_norm)
    true_theta = np.arcsin(np.sin(apparent_theta) / refraction_index)
    true_vector = np.vstack([true_vector_norm * np.sin(true_theta) * np.sin(gisement),
                             true_vector_norm * np.sin(true_theta) * np.cos(gisement),
                             true_vector_norm * np.cos(true_theta)])
    return np.transpose(true_vector)


class PyC2C(object):
    def __init__(self, compared, reference, dim=3,neighbors=1, cores=20):
        if type(compared) == las.lasdata and type(reference) == las.lasdata:
            self.compared, self.reference=compared, reference
            self.compared_file, self.reference_file = None, None

        else:
            self.compared_file, self.reference_file = compared, reference
            self.compared = las.read(compared, True)
            self.reference = las.read(reference)
        
        if dim not in [2,3]:
            raise Exception("dim must be in [2,3]")
        self.compute(self.compared.XYZ,self.reference.XYZ,dim,cores,neighbors)

    def compute(self, comp, ref, dim, nb_cores, neigh):
        tree=cKDTree(ref[:,0:dim],leafsize=250)
        index=tree.query(comp[:,0:dim],k=neigh,p=2,n_jobs=nb_cores)[1]

        if neigh==1:
            diff=comp-ref[index]
        else:
            diff=comp-np.mean(ref[index],axis=1)
            
        self.dist_plani=np.linalg.norm(diff[:,0:2],ord=2,axis=1)
        self.dist_alti=diff[:,2]

    def get_distances(self):
        return {"dist_plani":self.dist_plani,"dist_alti":self.dist_alti}
    
    def save(self):
        if self.compared_file is not None:
            extra1=[{"name":"dist_plani","type":'float32',"data":self.dist_plani},
                    {"name":"dist_alti","type":'float32',"data":self.dist_alti}]
            extra=extra1+[{"name":i,"type":getattr(self.compared,i).dtype.name,"data":getattr(self.compared,i)} for i in self.compared.metadata['extraField']]
            las.WriteLAS(self.compared_file[0:-4] + "_C2C.laz", self.compared, extra_fields=extra)


class Alphashape(object):
    def __init__(self, points2D, alpha=100):
        self.points = points2D
        self.alpha = alpha
        self.compute_alphashape()
        
    def compute_alphashape(self):
        # More you decrease alpha factor, greater the constraint will be on alphashape
        # More you increase alpha factor, more the alphashape will be a convex hull
        print("[Alphashape] Delaunay triangulation...",end='')
        tri=scp.Delaunay(self.points)
        nbrVertices = len(tri.vertices)
        print("done !")

        print(f'[Alphashape] Cleaning {nbrVertices} triangles :')
        displayer = misc.Timing(nbrVertices, 20)
        
        edges=set()
        edge_points=[]
        for i in range(0,nbrVertices):
            msg=displayer.timer(i)
            if msg is not None:
                print(f'[Alphashape] {msg}')
                
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
        print(f'[Alphashape] Cleaning {nbrVertices} triangles : done !')

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


def compute_dbscan(filepath, maxdist=1, minsamples=5):
    """make Scikit-Learn DBSCAN clustering
    (see docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

    Args:
        filepath (str): path to LAS file
        mindist (int, optional): Maximum distance between two samples. Defaults to 1.
        minsamples (int, optional): Minimum number of samples in each cluster. Defaults to 5.
    """
    data= las.read(filepath)
    model=DBSCAN(eps=maxdist,min_samples=minsamples,algorithm='kd_tree',leaf_size=1000,n_jobs=46).fit(data.XYZ)
    
    if len(np.unique(model.labels_))>1:
        extra=[(("labels","int16"),model.labels_)]
        print(f'Number of clusters : {len(np.unique(model.labels_))-1}')
        las.WriteLAS(filepath[0:-4] + "_DBSCAN.laz", data, extraField=extra)
    else:
        print("DBSCAN find only 1 cluster !")


def compute_density(points, core_points=[], radius=1, p_norm=2):
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
    tab_fwf,metadata_fwf= las.read(workspace + fichier, "fwf")
    tab_extra,metadata_extra= las.read(workspace + fichier[0:-4] + "_extra.laz", "standard", True)
    names_fwf=metadata_fwf['col_names']
    names_extra=metadata_extra['col_names']
    
    controle=np.sqrt((tab_fwf[:,0]-tab_extra[:,0])**2+(tab_fwf[:,1]-tab_extra[:,1])**2+(tab_fwf[:,2]-tab_extra[:,2])**2)
    try: assert(all(controle<0.003))
    except:
        raise Exception("LAS_FWF file and LAS file don't match exactly!\nPlease check your files...")

    dist_Z = tab_extra[:,names_extra.index('c2c_absolute_distances_(z)')]
    dist_plani = np.sqrt(np.power(tab_extra[:,names_extra.index('c2c_absolute_distances_(x)')],2)+np.power(tab_extra[:,names_extra.index('c2c_absolute_distances_(y)')],2))
    num = names_fwf.index("wave_packet_desc_index")
    tab_tot = np.hstack([tab_extra[:,0:-4],tab_fwf[:,num::],np.reshape(dist_Z,(len(dist_Z),1)),np.reshape(dist_plani,(len(dist_plani),1))])
    names_tot = names_extra[0:-4]+names_fwf[num::]+['depth','distance_H']
    return tab_tot, names_tot, metadata_fwf['vlrs']


def select_pairs_overlap(filepath, shifts):
    files = glob.glob(filepath)
    polygons = []
    num_list = []
    for file in files:
        data = las.read(file)
        head, tail = os.path.split(file)
        num_list += [str(tail[shifts[0]: shifts[0] + shifts[1]])]
        pca_pts = PCA(n_components=2, svd_solver='full')
        dat_new = pca_pts.fit_transform(data.XYZ[:, 0:2])
        del data
        
        boundaries = np.array([[min(dat_new[:, 0]), min(dat_new[:, 1])],
                               [min(dat_new[:, 0]), max(dat_new[:, 1])],
                               [max(dat_new[:, 0]), max(dat_new[:, 1])],
                               [max(dat_new[:, 0]), min(dat_new[:, 1])]])
        new_boundaries = pca_pts.inverse_transform(boundaries)
        polygons += [Polygon(new_boundaries)]

    comparison = {}
    n_polygons = len(polygons)
    overlaps = []
    for idx_a in range(0, n_polygons - 1):
        polygon_a = polygons[idx_a]
        listing = []
        for idx_b in range(idx_a + 1, n_polygons):
            polygon_b = polygons[idx_b]
            if polygon_a.overlaps(polygon_b):
                diff = polygon_a.difference(polygon_b)
                remaining = diff.area / polygon_a.area
                if remaining < 0.9:
                    listing += [num_list[idx_b]]
                    overlap_pct = (1 - remaining) * 100
                    overlaps.append((num_list[idx_a], num_list[idx_b], overlap_pct))
                    print(f'overlap area between line {num_list[idx_a]} and {num_list[idx_b]} = {overlap_pct:.2f} %')

        if len(listing) > 0:
            comparison[num_list[idx_a]] = listing

    return comparison, overlaps


def write_kml(filepath, names, descriptions, coordinates):
    try:
        assert(len(names) == len(descriptions) and len(names) == len(coordinates) and len(descriptions) == len(coordinates))
    except:
        print("Different sizes for names, description and coords !!")
    file = simplekml.Kml()
    for i in names:
        file.newpoint(name=i, description=descriptions[names.index(i)], coords=[coordinates[names.index(i)]])
    file.save(filepath)
    return True
