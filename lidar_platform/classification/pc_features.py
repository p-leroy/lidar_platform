"""
Created on Fri Apr 15 09:11:06 2022

@author: Mathilde Letard
"""
import numpy as np

import open3d as o3d
import scipy
#from ..cc_3dmasc import load_sbf_features

import sys, os
sys.path.insert(0, os.path.abspath(r'C:\DEV'))
import lidar_platform as pl
from lidar_platform.classification.cc_3dmasc import load_sbf_features


def l2dist(p1,p2):
    """
    Get L2 distance between two 3D points.
    
    Parameters
    ----------
    p1 : numpy array 1x3
        XYZ coordinates of point 1.
    p2 : numpy array 1x3
        XYZ coordinates of point 2.

    Returns
    -------
    float
        L2 distance between the two points.
    """
    a = (p2[0]-p1[0])**2
    b = (p2[1]-p1[1])**2
    c = (p2[2]-p1[2])**2
    return np.sqrt(a+b+c)


def stdl2dist(pref,pts):
    """
    Get standard deviation of L2 distance between 3D points.
    
    Parameters
    ----------
    pref : numpy array 1x3
        XYZ coordinates of point 1.
    pts : numpy array nx3
        XYZ coordinates of n points.

    Returns
    -------
    float
        standard deviation of L2 distance between the points.

    """
    a = (pts[:,0]-pref[0])**2
    b = (pts[:,1]-pref[1])**2
    c = (pts[:,2]-pref[2])**2
    total = np.sqrt(a+b+c)
    return np.std(total)


def medl1dist(pts,pref):
    """
    Get median L1 distance between 3D points.
    
    Parameters
    ----------
    p1 : numpy array 1x3
        XYZ coordinates of point 1.
    p2 : numpy array nx3
        XYZ coordinates of n points.

    Returns
    -------
    float
        median L1 distance between the points.

    """
    dx = np.abs(pts[:,0]-pref[0])
    dy = np.abs(pts[:,1]-pref[1])
    dz = np.abs(pts[:,2]-pref[2])
    d = dx+dy+dz
    return np.median(d)


def moment(p, pvect, value, order):
    """
    Compute statistical moment of a point's neighborhood about an eigenvector at a given order.

    Parameters
    ----------
    p : numpy array 1x3
        XYZ coordinates of core point.
    pvect : numpy array nx3
        XYZ coordinates of n neighbor points.
    value : numpy array 1x3
        eigenvector of the covariance matrix of the spherical neighborhood.
    order :  int
        order of the statistical moment

    Returns
    -------
    float
        statistical moment value.

    """
    d = []
    for pt in pvect:
        dot = np.dot(l2dist(pt, p), value)
        d.append(dot ** order)
    return np.abs(np.sum(d)) / pvect.shape[0]


def unit_vector(vector):
    """
    Returns the unit vector of the vector.

    Parameters
    ----------
    vector : numpy array 1x3
        input vector.

    Returns
    -------
    numpy array 1x3
        unit vector.

    """
    return vector / np.linalg.norm(vector)


def angle(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'

    Parameters
    ----------
    v1 : numpy array 1x3
        input vector 1.
    v2 : numpy array 1x3
        input vector 2

    Returns
    -------
    float
        angle between vectors v1 and v2.

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))    


def get_classical3Dfeatures(corepts, corepts_labels, cloud, scales, suffix):
    """
    Compute M descriptive features of a given point cloud.

    Parameters
    ----------
    corepts : array (Nx3)
        XYZ coordinates of the core points.
    cloud : array (Nx3)
        XYZ coordinates of the N points constituting the point cloud.
    r : float
        radius for the search around each point of the cloud.
    suffix : str
        string to put at the end of the default file name.

    Returns 
    -------
    pc_feats : dictio
        a dictionary containing the each feature's value for each point (keys = feature names).
    The point cloud with M features for each point (numpy array of size NxM)

    """
    pts = cloud[:, [0, 1, 2]]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)        
    pc_tree = o3d.geometry.KDTreeFlann(pc)    
    corepts = corepts[:, [0, 1, 2]]
    corepc = o3d.geometry.PointCloud()
    corepc.points = o3d.utility.Vector3dVector(corepts)
    n_p = corepts.shape[0]

    pc_feats = {'labels': np.array(corepts_labels)}

    for s in scales:
        pc_feats['moment_1_1@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['moment_1_2@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['moment_1_3@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['moment_2_1@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['moment_2_2@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['moment_2_3@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['sum_eigenvalues@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['omnivariance@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['eigen_entropy@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['surface_variation@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['verticality_1@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['verticality_2@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['curv_change@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['vertical_moment_1@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['vertical_moment_2@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['knn_max_h_diff@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['knn_h_std@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['normal_deviation@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['plane_residuals@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['dist_to_plane@' + str(s)] = np.zeros((n_p, 1)) - 999
        pc_feats['deviation_variance@' + str(s)] = np.zeros((n_p, 1)) - 999
    
    print('Computing features on '+str(corepts.shape[0])+' points with '+str(len(scales))+' scales...')

    for i in range(corepts.shape[0]):
        for s in scales:
            #print(s)
            p = corepc.points[i]
            krad, idx_sphere, _ = pc_tree.search_radius_vector_3d(p, s/2)
            sphere = np.asarray(pc.points)[idx_sphere, :]
            if len(idx_sphere) > 0:
                z_sphere = sphere[:, 2]
                sphere_pc = o3d.geometry.PointCloud()
                sphere_pc.points = o3d.utility.Vector3dVector(sphere)
                cov_mat = sphere_pc.compute_mean_and_covariance()[1]
                w, v = np.linalg.eig(cov_mat)
                l1 = w[0]
                l2 = w[1]
                l3 = w[2]
                e1 = v[0]
                e2 = v[1]
                e3 = v[2]
                ez = np.array([0, 0, 1])
                pc_feats['sum_eigenvalues@'+str(s)][i] = np.sum(w)
                pc_feats['surface_variation@'+str(s)][i] = l3 / (np.sum(w))
                pc_feats['curv_change@'+str(s)][i] = l3 / (np.sum(w))
                pc_feats['knn_h_std@'+str(s)][i] = np.std(z_sphere)
                pc_feats['knn_max_h_diff@'+str(s)][i] = np.max(z_sphere - p[2])
                pc_feats['omnivariance@'+str(s)][i] = (l1 * l2 * l3) ** (1/3)
                pc_feats['eigen_entropy@'+str(s)][i] = l1 * np.log(l1) + l2 * np.log(l2) + l3 * np.log(l3)
                pc_feats['moment_1_1@'+str(s)][i] = moment(p, sphere, e1, 1)
                pc_feats['moment_1_2@'+str(s)][i] = moment(p, sphere, e2, 1)
                pc_feats['moment_1_3@'+str(s)][i] = moment(p, sphere, e3, 1)
                pc_feats['moment_2_1@'+str(s)][i] = moment(p, sphere, e1, 2)
                pc_feats['moment_2_2@'+str(s)][i] = moment(p, sphere, e2, 2)
                pc_feats['moment_2_3@'+str(s)][i] = moment(p, sphere, e3, 2)
                pc_feats['verticality_1@'+str(s)][i] = np.abs((np.pi / 2) - angle(e1, ez))
                pc_feats['verticality_2@'+str(s)][i] = np.abs((np.pi / 2) - angle(e3, ez))
                pc_feats['vertical_moment_1@'+str(s)][i] = moment(p, sphere, ez, 1)
                pc_feats['vertical_moment_2@'+str(s)][i] = moment(p, sphere, ez, 2)
                pc_feats['normal_deviation@'+str(s)][i] = angle(e3, ez)
                angles = [angle(e1, ez), angle(e2, ez), angle(e3, ez)]
                pc_feats['deviation_variance@'+str(s)][i] = np.var(angles)
                A = np.c_[sphere[:, 0], sphere[:, 1], np.ones(sphere.shape[0])]
                C, resid, _, _ = scipy.linalg.lstsq(A, sphere[:, 2])
                pc_feats['plane_residuals@'+str(s)][i] = np.mean(resid)
                pt_plane_coords = [p[0], p[1], C[0] * p[0] + C[1] * p[1] + C[2]]
                pc_feats['dist_to_plane@'+str(s)][i] = l2dist(p, pt_plane_coords)

    np.save('classical3DfeaturesPC_'+suffix+'.npy', pc_feats)
    return pc_feats

trainpts = load_sbf_features(r'C:\Users\mathi\Desktop\Ain_calculs\test_A11.sbf', r'C:\Users\mathi\Desktop\3DMASC_params.txt', labels=True, coords=True)
#testpts = load_sbf_features(r'C3_test_5cl_2000samples.sbf', r'3DMASC_params.txt', labels=True)
#cloud = pl.classification.cc_3dmasc.load_sbf_features(r'C:\Users\mathi\Desktop\Ain_calculs\NIR_buffer_PC.sbf', r'C:\Users\mathi\Desktop\3DMASC_params.txt', coords=True)
#np.save('nir_Ain.npy', cloud['coords'])
np.save('test11A.npy', trainpts['coords'])
#print("done reading")
#scales = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5]
#get_classical3Dfeatures(trainpts['coords'], trainpts['labels'], cloud['coords'], scales, 'suffix')