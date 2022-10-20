# coding: utf-8
# Baptiste Feldmann

import numpy as np
from scipy.spatial import cKDTree

from joblib import Parallel, delayed

from . import classification


class ComputeFeatures(object):
    def __init__(self, pc1, pcx, scales):
        self.pc1 = pc1
        self.pcx = pcx
        self.scales = scales
        self.maxCores = 25
        self.minmaxSizePack = [30000, 35000]
        
    def __pack_data(self, data):
        data_pack = []
        data_size = len(data[:, 0])
        begin_end = []
        if self.minmaxSizePack[1] * self.maxCores > data_size:
            if self.minmaxSizePack[0] * self.maxCores > data_size:
                size_pack = self.minmaxSizePack[0]
            else:
                size_pack = int(data_size / self.maxCores)
        else:
            size_pack = self.minmaxSizePack[1]
        
        if size_pack >= data_size:
            data_pack += [data]
            begin_end += [[0, data_size]]
        else:
            pack_list = np.arange(0, data_size, size_pack)
            for i in pack_list[0: -1]:
                data_pack += [data[i:i + size_pack, :]]
                begin_end += [[i, i + size_pack]]
                
            if pack_list[-1] < (data_size-1):
                data_pack += [data[pack_list[-1]:data_size]]
                begin_end += [[pack_list[-1], data_size]]
            
        return data_pack, begin_end

    def __compute_features_1p(self, list_of_points, scale):
        data_list = classification.compute_features_1p(self.tree, list_of_points, scale)
        return data_list

    def compute_all_scales(self):
        coords_pcx = self.pcx[:, 0:3]
        pack_pcx, list_pack = self.__pack_data(coords_pcx)
        if len(pack_pcx) >= self.maxCores:
            nb_cores = self.maxCores
        else:
            nb_cores = len(pack_pcx)

        print("build KD Tree...", end="\r")
        coords_pc1 = np.array(self.pc1[:, 0:3])
        self.tree = cKDTree(coords_pc1, leafsize=100000)
        print("Done !")
        for scale in self.scales:
            print("Scales analyzed : " + str(scale))
            # listing_pts_pc1=kdtree_pc1.query_ball_point(np.array(pcx[:,0:3],order='C'),r=scale,n_jobs=-1)
            # descriptors=[Parallel(n_jobs=nb_cores,verbose=2)(delayed(compute_features_1p)(listing_pts_pc1[list_pack[i][0]:list_pack[i][1]],pc1,pack_pcx[i],scale) for i in range(0,len(pack_pcx)))]
            descriptors = [Parallel(n_jobs=nb_cores, verbose=3)
                           (delayed(self.__compute_features_1p)(pack_pcx[i], scale)
                            for i in range(0, len(pack_pcx)))]
            # descriptors=[compute_features_1p(pc1[listing_pts_pc1[ind],:]) for ind in range(0,len(pcx[:,0]))]
            # descriptors=np.array(Parallel(n_jobs=50,verbose=1)(delayed(compute_features_1p)(ind,pc1[listing_pts_pc1[ind],:]) for ind in range(0,len(pcx[:,0]))))

        return np.concatenate(descriptors, axis=0)
