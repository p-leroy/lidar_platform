# coding: utf-8
# Baptiste Feldmann

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from joblib import Parallel, delayed


class ComputeFeatures(object):
    def __init__(self,PC1,PCX,scales):
        self.pc1=PC1
        self.pcx=PCX
        self.scales=scales
        self.maxCores=25
        self.minmaxSizePack=[30000,35000]
        
    def __packData(self,data):
        dataPack=[]
        sizeData=len(data[:,0])
        beginEnd=[]
        if self.minmaxSizePack[1]*self.maxCores>sizeData:
            if self.minmaxSizePack[0]*self.maxCores>sizeData:
                size_pack=self.minmaxSizePack[0]
            else:
                size_pack=int(sizeData/self.maxCores)
        else:
            size_pack=self.minmaxSizePack[1]
        
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
  
    def __compute_acp(self,coords):
        pca_pts=PCA(n_components=3,svd_solver='full')
        pca_pts.fit(coords)
        ratio=pca_pts.explained_variance_ratio_
        return ratio
        #normal=pca_pts.components_[2]
        #slope=90 - np.absolute(np.degrees(math.atan2(normal[2], np.sqrt((normal[0]**2)+(normal[1]**2)))))
        #return ratio,slope

    def __compute_features_1p(self,listCoordsPcx,scale):
        listData=[]
        listing_pts=self.tree.query_ball_point(listCoordsPcx,r=scale,n_jobs=-1)
        for i in range(0,len(listCoordsPcx)):
            select_pts=self.pc1[listing_pts[i],:]
            if len(select_pts[:,0])>3:
                ratio_pts=self.__compute_acp(select_pts[:,0:3])
            else:
                ratio_pts=np.array([11/18,5/18,1/9])
            listData+=[[ratio_pts[0],ratio_pts[1],ratio_pts[2]]]
        return listData

    def compute_all_scales(self):
        coords_pcx=self.pcx[:,0:3]
        pack_pcx,list_pack=self.__packData(coords_pcx)
        if len(pack_pcx)>=self.maxCores:
            nb_cores=self.maxCores
        else:
            nb_cores=len(pack_pcx)

        print("build KD Tree...",end="\r")
        coords_pc1=np.array(self.pc1[:,0:3])
        self.tree=cKDTree(coords_pc1,leafsize=100000)
        print("Done !")
        for scale in self.scales:
            print("Scales analyzed : "+str(scale))         
            #listing_pts_pc1=kdtree_pc1.query_ball_point(np.array(PCX[:,0:3],order='C'),r=scale,n_jobs=-1)
            #descriptors=[Parallel(n_jobs=nb_cores,verbose=2)(delayed(compute_features_1p)(listing_pts_pc1[list_pack[i][0]:list_pack[i][1]],PC1,pack_pcx[i],scale) for i in range(0,len(pack_pcx)))]
            descriptors=[Parallel(n_jobs=nb_cores,verbose=3)(delayed(self.__compute_features_1p)(pack_pcx[i],scale) for i in range(0,len(pack_pcx)))]
            #descriptors=[compute_features_1p(PC1[listing_pts_pc1[ind],:]) for ind in range(0,len(PCX[:,0]))]
            #descriptors=np.array(Parallel(n_jobs=50,verbose=1)(delayed(compute_features_1p)(ind,PC1[listing_pts_pc1[ind],:]) for ind in range(0,len(PCX[:,0]))))
        return np.concatenate(descriptors,axis=0)
