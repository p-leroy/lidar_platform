# coding: utf-8
# Baptiste Feldmann
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
from scipy.spatial import cKDTree
from scipy.stats import mode
import math
import os
from joblib import Parallel,delayed,parallel_backend
import pickle
import copy

def pack_data(data):
    minmaxSizePack=[50000,100000]
    maxCores=45
    dataPack=[]
    sizeData=len(data[:,0])
    beginEnd=[]
    if minmaxSizePack[1]*maxCores>sizeData:
        if minmaxSizePack[0]*maxCores>sizeData:
            size_pack=minmaxSizePack[0]
        else:
            size_pack=int(sizeData/maxCores)
    else:
        size_pack=minmaxSizePack[1]
    
    if size_pack>=sizeData:
        dataPack+=[data]
        beginEnd+=[[0,sizeData]]
    else:
        listPack=np.arange(0,sizeData,size_pack)
        for i in listPack[0:-1]:
            dataPack+=[data[i:i+size_pack,:]]
            beginEnd+=[[i,i+size_pack]]
            
        if listPack[-1]<(sizeData-1):
            dataPack+=[data[listPack[-1]:sizeData]]
            beginEnd+=[[listPack[-1],sizeData]]
        
    return dataPack,beginEnd   

def compute_acp(coords):
    pca_pts=PCA(n_components=3,svd_solver='full')
    pca_pts.fit(coords)
    ratio=pca_pts.explained_variance_ratio_
    return ratio
    #normal=pca_pts.components_[2]
    #slope=90 - np.absolute(np.degrees(math.atan2(normal[2], np.sqrt((normal[0]**2)+(normal[1]**2)))))
    #return ratio,slope

def compute_features_1p(tree,PC1,listCoordsPcx,scale):
    listData=[]
    for i in range(0,len(listCoordsPcx)):
        listing_pts=tree.query_ball_point(listCoordsPcx[i],r=scale)
        select_pts=PC1[listing_pts,:]
        if len(select_pts[:,0])>3:
            ratio_pts=compute_acp(select_pts[:,0:3])
        else:
            ratio_pts=np.array([11/18,5/18,1/9])
        listData+=[[ratio_pts[0],ratio_pts[1],ratio_pts[2]]]
    return listData

def compute_features_all_scales(PC1,PCX,scales):
    max_jobs=45
    print("build KD Tree...",end="\r")
    coords_pc1=np.array(PC1[:,0:3])
    kdtree_pc1=cKDTree(coords_pc1,leafsize=1000)
    print("Done !")
    coords_pcx=PCX[:,0:3]
    pack_pcx,list_pack=pack_data(coords_pcx)
    if len(pack_pcx)>=max_jobs:
        nb_cores=max_jobs
    else:
        nb_cores=len(pack_pcx)
   
    features_scales=np.array([])
    for scale in scales:
        print("Scales analyzed : "+str(scale))
        #listing_pts_pc1=kdtree_pc1.query_ball_point(np.array(PCX[:,0:3],order='C'),r=scale,n_jobs=-1)
        #descriptors=[Parallel(n_jobs=nb_cores,verbose=2)(delayed(compute_features_1p)(listing_pts_pc1[list_pack[i][0]:list_pack[i][1]],PC1,pack_pcx[i],scale) for i in range(0,len(pack_pcx)))]
        descriptors=[Parallel(n_jobs=nb_cores,verbose=3)(delayed(compute_features_1p)(copy.deepcopy(kdtree_pc1),PC1,pack_pcx[i],scale) for i in range(0,len(pack_pcx)))]
        #descriptors=[compute_features_1p(PC1[listing_pts_pc1[ind],:]) for ind in range(0,len(PCX[:,0]))]
        #descriptors=np.array(Parallel(n_jobs=50,verbose=1)(delayed(compute_features_1p)(ind,PC1[listing_pts_pc1[ind],:]) for ind in range(0,len(PCX[:,0]))))
    return np.concatenate(descriptors,axis=0)
##        if not all(np.sort(descriptors[:,0])==descriptors[:,0]):
##            descriptors=descriptors[descriptors[:,0].argsort()]
##            
##        features_scales=np.hstack([features_scales,np.array(descriptors)])
##    print(" Done !")
##    return features_scales

def compute_features_for_one_point(indice,cloud_C3,cloud_C2,names):
    if len(cloud_C3[:,0])>3:
        ratio_pts,slope_pts=compute_acp(cloud_C3[:,0:3])
    else:
        ratio_pts=np.array([11/18,5/18,1/9])
        slope_pts = 0
    
    #Calcul des descripteurs C3
    std_height_pts = np.std(cloud_C3[:, 2])
    zrange=np.max(cloud_C3[:,2])-np.min(cloud_C3[:,2])
    median_intensity_pts = np.median(cloud_C3[:,np.where(names=='intensity')[0][0]])
    std_intensity_pts = np.std(cloud_C3[:,np.where(names=='intensity')[0][0]])
    #density=len(cloud_C3[:,0])

    if len(cloud_C2[:,0])>=1:
        #Calcul des descripteurs C2
        std_height_C2=np.std(cloud_C2[:,2])
        rapp_height=np.median(cloud_C3[:,2])/np.median(cloud_C2[:,2])
        #zrange_C2=np.max(cloud_C2[:,2])-np.min(cloud_C2[:,2])
        std_intensite_C2=np.std(cloud_C2[:,np.where(names=='intensity')[0][0]])
        rapp_intensite=median_intensity_pts/np.median(cloud_C2[:,np.where(names=='intensity')[0][0]])
        #median_intensite_C2=np.median(cloud_C2[:,np.where(names=='intensity')[0][0]])
        #nbr_pts_C2=len(cloud_C2[:,0])
        #class_C2=mode(cloud_C2[:,np.where(names=="classification")[0][0]])[0][0]
    else:
        #Calcul des descripteurs C2
        std_height_C2=-1
        rapp_height=0
        #zrange_C2=-1
        std_intensite_C2=-1
        rapp_intensite=0
        #median_intensite_C2=-1
        #nbr_pts_C2=0
        #class_C2=-1

    #Descripteurs totals
    #return [indice,ratio_pts[0],ratio_pts[1],ratio_pts[2],slope_pts,std_height_pts,zrange,median_intensity_pts,std_intensity_pts,density,std_height_C2,zrange_C2,std_intensite_C2,median_intensite_C2,nbr_pts_C2,class_C2]
    return [indice,ratio_pts[0],ratio_pts[1],ratio_pts[2],slope_pts,std_height_pts,zrange,median_intensity_pts,std_intensity_pts,std_height_C2,rapp_height,std_intensite_C2,rapp_intensite]

def compute_features_1pt_1cloud(cloud_C3,names):
    if len(cloud_C3[:,0])>3:
        ratio_pts,slope_pts=compute_acp(cloud_C3[:,0:3])
    else:
        ratio_pts=np.array([11/18,5/18,1/9])
        slope_pts = 0
    
    #Calcul des descripteurs C3
    std_height_pts = np.std(cloud_C3[:, 2])
    zrange=np.max(cloud_C3[:,2])-np.min(cloud_C3[:,2])
    median_intensity_pts = np.median(cloud_C3[:,np.where(names=='intensity')[0][0]])
    std_intensity_pts = np.std(cloud_C3[:,np.where(names=='intensity')[0][0]])
    density=len(cloud_C3[:,0])

    #Descripteurs totals
    return [ratio_pts[0],ratio_pts[1],ratio_pts[2],slope_pts,std_height_pts,zrange,median_intensity_pts,std_intensity_pts,density]

def compute_features_for_all_scales(workspace,pts_cloud,column_names,core_points,scales):
    print("build KD Tree...",end='\r')
    coords_C3=pts_cloud[:,0:3]
    kdtree_C3=cKDTree(coords_C3,leafsize=10000)
##    coords_C2=C2_cloud[:,0:3]
##    kdtree_C2=cKDTree(coords_C2,leafsize=10000)
    print("OK !")
    #names_features=["ratio1_C2","ratio2_C2","slopes_C2","std_z_C2","mean_intensity_C2","std_intensity_C2"]
    features_scales=np.hstack([core_points[:,column_names=='depth_surf'],core_points[:,column_names=='dist_plani_topo']])
    print("Scales analyzed : ",end='\r')
    for scale in scales:
        print(scale,end=" \r")
        listing_pts_C3=kdtree_C3.query_ball_point(core_points[:,0:3],r=scale,n_jobs=-1)
        #listing_pts_C2=kdtree_C2.query_ball_point(core_points[:,0:3],r=scale,n_jobs=-1)
        #names_features=["mean_dist_z","ratio1_C2","ratio2_C2","slopes_C2","std_z_C2","mean_intensity_C2","std_intensity_C2"]
        #descriptors=np.array(Parallel(n_jobs=56)(delayed(compute_features_for_one_point)(ind,pts_cloud[listing_pts_C3[ind],:],C2_cloud[listing_pts_C2[ind],:],column_names) for ind in range(0,len(core_points[:,0]))))
        descriptors=[compute_features_1pt_1cloud(pts_cloud[listing_pts_C3[ind],:],column_names) for ind in range(0,len(core_points[:,0]))]
##        descriptors=np.array(Parallel(n_jobs=30)(delayed(compute_features_1pt_1cloud)(ind,pts_cloud[listing_pts_C3[ind],:],column_names) for ind in range(0,len(core_points[:,0]))))
##        if not all(np.sort(descriptors[:,0])==descriptors[:,0]):
##            descriptors=descriptors[descriptors[:,0].argsort()]
            
        features_scales=np.hstack([features_scales,np.array(descriptors)])
    print(" done !")
    return features_scales

def RandomForest_training(data, labels, n_estimators=100, weight=None):
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

def RandomForest_classifier(model, data,labels=[]):
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

def classif_report(labels_true,labels_pred):
    labels=np.unique(labels_true)
    tab=metrics.confusion_matrix(labels_true,labels_pred,labels)
    affiche="Confusion matrix :\n\t"
    for i in labels:
        affiche+=str(i)+"\t"
    affiche+="\n"
    for i in range(0,len(labels)):
        affiche+=str(labels[i])+"\t"
        for c in range(0,len(labels)):
            affiche+=str(tab[i,c])+"\t"
        affiche+="\n"
    print(affiche)
    print()
    test=labels_true==labels_pred
    val_true=len([i for i,X in enumerate(test) if X])
    print("Pourcentage valeur Vrai : %.1f%%" %(val_true/len(test)*100))
    print("indice Kappa : "+str(metrics.cohen_kappa_score(labels_true,labels_pred)))
    print(metrics.classification_report(labels_true,labels_pred,labels))
