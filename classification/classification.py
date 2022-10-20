# coding: utf-8
# Baptiste Feldmann

import copy

import numpy as np
from scipy.spatial import cKDTree

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

from joblib import Parallel, delayed


def pack_data(data):
    minmaxSizePack = [50000, 100000]
    maxCores = 45
    data_pack = []
    sizeData = len(data[:, 0])
    begin_end = []
    if minmaxSizePack[1] * maxCores > sizeData:
        if minmaxSizePack[0] * maxCores > sizeData:
            size_pack = minmaxSizePack[0]
        else:
            size_pack = int(sizeData / maxCores)
    else:
        size_pack = minmaxSizePack[1]
    
    if size_pack >= sizeData:
        data_pack += [data]
        begin_end += [[0, sizeData]]
    else:
        pack_list = np.arange(0,sizeData,size_pack)
        for i in pack_list[0:-1]:
            data_pack += [data[i:i + size_pack, :]]
            begin_end += [[i, i + size_pack]]
            
        if pack_list[-1] < (sizeData - 1):
            data_pack += [data[pack_list[-1]: sizeData]]
            begin_end += [[pack_list[-1], sizeData]]
        
    return data_pack, begin_end


def compute_pca(coords):
    pca_pts = PCA(n_components=3, svd_solver='full')
    pca_pts.fit(coords)
    ratio = pca_pts.explained_variance_ratio_
    return ratio
    #normal=pca_pts.components_[2]
    #slope=90 - np.absolute(np.degrees(math.atan2(normal[2], np.sqrt((normal[0]**2)+(normal[1]**2)))))
    #return ratio,slope


def compute_features_1p(tree, pc1, list_of_points, scale):
    data_list = []
    lists_of_neighbors = tree.query_ball_point(list_of_points, r=scale, workers=-1)
    for list_of_neighbors in lists_of_neighbors:
        select_pts = pc1[list_of_neighbors, :]
        if len(select_pts[:, 0]) > 3:
            ratio_pts = compute_pca(select_pts[:, 0:3])
        else:
            ratio_pts = np.array([11 / 18, 5 / 18, 1 / 9])
        data_list += [[ratio_pts[0], ratio_pts[1], ratio_pts[2]]]
    return data_list


def compute_features_all_scales(pc1, pcx, scales):
    max_jobs = 45
    print("build KD Tree...", end="\r")
    coords_pc1 = np.array(pc1[:, 0:3])
    kdtree_pc1 = cKDTree(coords_pc1, leafsize=1000)
    print("Done !")
    coords_pcx = pcx[:, 0:3]
    pack_pcx, list_pack = pack_data(coords_pcx)
    if len(pack_pcx) >= max_jobs:
        nb_cores = max_jobs
    else:
        nb_cores = len(pack_pcx)
   
    features_scales = np.array([])
    for scale in scales:
        print("Scales analyzed : " + str(scale))
        # listing_pts_pc1=kdtree_pc1.query_ball_point(np.array(pcx[:,0:3],order='C'),r=scale,n_jobs=-1)
        # descriptors=[Parallel(n_jobs=nb_cores,verbose=2)(delayed(compute_features_1p)
        # (listing_pts_pc1[list_pack[i][0]:list_pack[i][1]],pc1,pack_pcx[i],scale) for i in range(0,len(pack_pcx)))]
        descriptors = [Parallel(n_jobs=nb_cores,verbose=3)
                       (delayed(compute_features_1p)(copy.deepcopy(kdtree_pc1), pc1, pack_pcx[i], scale)
                        for i in range(0, len(pack_pcx)))]
        # descriptors=[compute_features_1p(pc1[listing_pts_pc1[ind],:]) for ind in range(0,len(pcx[:,0]))]
        # descriptors=np.array(Parallel(n_jobs=50,verbose=1)(delayed(compute_features_1p)
        # (ind,pc1[listing_pts_pc1[ind],:]) for ind in range(0,len(pcx[:,0]))))
    return np.concatenate(descriptors, axis=0)
#        if not all(np.sort(descriptors[:,0])==descriptors[:,0]):
#            descriptors=descriptors[descriptors[:,0].argsort()]
#
#        features_scales=np.hstack([features_scales,np.array(descriptors)])
#    print(" Done !")
#    return features_scales


def compute_features_for_one_point(index, cloud_c3, cloud_c2, names):
    if len(cloud_c3[:, 0]) > 3:
        ratio_pts,slope_pts = compute_pca(cloud_c3[:, 0:3])
    else:
        ratio_pts = np.array([11 / 18, 5 / 18, 1 / 9])
        slope_pts = 0
    
    # C3 descriptors calculations
    std_height_pts = np.std(cloud_c3[:, 2])
    z_range = np.max(cloud_c3[:, 2]) - np.min(cloud_c3[:, 2])
    median_intensity_pts = np.median(cloud_c3[:, np.where(names == 'intensity')[0][0]])
    std_intensity_pts = np.std(cloud_c3[:, np.where(names == 'intensity')[0][0]])
    # density=len(cloud_c3[:,0])

    if len(cloud_c2[:, 0]) >= 1:
        # C2 descriptors calculation
        std_height_c2 = np.std(cloud_c2[:, 2])
        rapp_height = np.median(cloud_c3[:, 2]) / np.median(cloud_c2[:, 2])
        # zrange_C2=np.max(cloud_c2[:,2])-np.min(cloud_c2[:,2])
        std_intensity_c2 = np.std(cloud_c2[:, np.where(names == 'intensity')[0][0]])
        intensity_ratio = median_intensity_pts / np.median(cloud_c2[:, np.where(names == 'intensity')[0][0]])
        # median_intensite_C2=np.median(cloud_c2[:,np.where(names=='intensity')[0][0]])
        # nbr_pts_C2=len(cloud_c2[:,0])
        # class_C2=mode(cloud_c2[:,np.where(names=="classification")[0][0]])[0][0]
    else:
        # C2 descriptors calculation
        std_height_c2 = -1
        rapp_height = 0
        # zrange_C2=-1
        std_intensity_c2 = -1
        intensity_ratio = 0
        # median_intensite_C2=-1
        # nbr_pts_C2=0
        # class_C2=-1

    # Descriptors
    # return [index,ratio_pts[0],ratio_pts[1],ratio_pts[2],
    # slope_pts,std_height_pts,z_range,median_intensity_pts,std_intensity_pts,density,std_height_C2,
    # zrange_C2,std_intensite_C2,median_intensite_C2,nbr_pts_C2,class_C2]
    return [index, ratio_pts[0], ratio_pts[1], ratio_pts[2],
            slope_pts, std_height_pts, z_range, median_intensity_pts, std_intensity_pts, std_height_c2,
            rapp_height, std_intensity_c2, intensity_ratio]


def compute_features_1pt_1cloud(cloud_C3, names):
    if len(cloud_C3[:,0]) > 3:
        ratio_pts, slope_pts = compute_pca(cloud_C3[:, 0:3])
    else:
        ratio_pts = np.array([11 / 18, 5 / 18, 1 / 9])
        slope_pts = 0
    
    # C3 descriptors computation
    std_height_pts = np.std(cloud_C3[:, 2])
    z_range = np.max(cloud_C3[:,2])-np.min(cloud_C3[:,2])
    median_intensity_pts = np.median(cloud_C3[:, np.where(names == 'intensity')[0][0]])
    std_intensity_pts = np.std(cloud_C3[:,np.where(names == 'intensity')[0][0]])
    density = len(cloud_C3[:, 0])

    # Descriptors
    return [ratio_pts[0], ratio_pts[1], ratio_pts[2],
            slope_pts, std_height_pts, z_range, median_intensity_pts, std_intensity_pts, density]


def compute_features_for_all_scales(workspace, pts_cloud, column_names, core_points, scales):
    print("build KD Tree...", end='\r')
    coords_C3 = pts_cloud[:, 0:3]
    kdtree_C3 = cKDTree(coords_C3, leafsize=10000)
    # coords_C2=C2_cloud[:,0:3]
    # kdtree_C2=cKDTree(coords_C2,leafsize=10000)
    print("OK !")
    # names_features=["ratio1_C2","ratio2_C2","slopes_C2","std_z_C2","mean_intensity_C2","std_intensity_C2"]
    features_scales = np.hstack([core_points[:, column_names == 'depth_surf'],
                                 core_points[:, column_names == 'dist_plani_topo']])
    print("Scales analyzed : ", end='\r')
    for scale in scales:
        print(scale, end=" \r")
        listing_pts_c3 = kdtree_C3.query_ball_point(core_points[:,0:3], r=scale, n_jobs=-1)
        # listing_pts_C2=kdtree_C2.query_ball_point(core_points[:,0:3],r=scale,n_jobs=-1)
        # names_features=["mean_dist_z","ratio1_C2","ratio2_C2","slopes_C2","std_z_C2","mean_intensity_C2","std_intensity_C2"]
        # descriptors=np.array(Parallel(n_jobs=56)(delayed(compute_features_for_one_point)
        # (ind,pts_cloud[listing_pts_C3[ind],:],C2_cloud[listing_pts_C2[ind],:],column_names)
        # for ind in range(0,len(core_points[:,0]))))
        descriptors = [compute_features_1pt_1cloud(pts_cloud[listing_pts_c3[ind], :], column_names)
                       for ind in range(0,len(core_points[:, 0]))]
        # descriptors=np.array(Parallel(n_jobs=30)(delayed(compute_features_1pt_1cloud)
        # (ind,pts_cloud[listing_pts_C3[ind],:],column_names) for ind in range(0,len(core_points[:,0]))))
        # if not all(np.sort(descriptors[:,0])==descriptors[:,0]):
        #     descriptors=descriptors[descriptors[:,0].argsort()]
            
        features_scales = np.hstack([features_scales, np.array(descriptors)])
    print(" done !")
    return features_scales


def random_forest_training(data, labels, n_estimators=100, weight=None):
    """Train a random forest

    Args:
        data: X
        labels: Y
        n_estimators: number of tree in RF (100 by default)
        max_features: number of features used to build a tree (the square root
                      of the number of all features by default)
        max_depth: max depth of tree (25 by default)
    """
    random_forest = RandomForestClassifier(n_estimators = n_estimators,
                                           criterion="gini",
                                           max_features="auto",
                                           max_depth=None,
                                           oob_score=True,
                                           n_jobs=-1,
                                           verbose=1,
                                           class_weight=weight)

    random_forest.fit(data, labels)
    return random_forest


def random_forest_classifier(model, data, labels=None):
    """Test the classifier

    Args:
        model: the trained random forest
        data: X
        labels: Y (optionnal)

    Returns:
        labels_predicts:
        confid_predict: the confidence
        error_rate: only if labels is not empty
        mean_confid: the average of confidence on good prediction, if labels is
                     not empty
    """
    labels_predict = model.predict(data)
    confid_predict = model.predict_proba(data)
    confid_predict = np.max(confid_predict, axis=1)

    if (len(labels) == np.shape(data)[0]):
        labels = labels.reshape(-1)
        foo = np.equal(labels, labels_predict)
        error_rate = np.count_nonzero(foo) / len(labels)
        confid_good = confid_predict[foo]
        mean_confid = np.mean(confid_good)
        return labels_predict, confid_predict, error_rate, mean_confid
    else:
        return labels_predict, confid_predict


def classification_report(labels_true, labels_pred):
    labels = np.unique(labels_true)
    tab = metrics.confusion_matrix(labels_true, labels_pred, labels)
    display = "Confusion matrix :\n\t"
    for i in labels:
        display += str(i) + "\t"
    display += "\n"
    for i in range(0, len(labels)):
        display += str(labels[i]) + "\t"
        for c in range(0, len(labels)):
            display += str(tab[i,c]) + "\t"
        display += "\n"
    print(display)
    print()
    test = labels_true == labels_pred
    val_true = len([i for i,X in enumerate(test) if X])
    print("Percents of true value: %.1f%%" % (val_true / len(test) * 100))
    print("Kappa coefficient: " + str(metrics.cohen_kappa_score(labels_true, labels_pred)))
    print(metrics.classification_report(labels_true, labels_pred, labels))
