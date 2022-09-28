import os

import numpy as np

from sklearn.cluster import DBSCAN

from tools import cc

idir = r'C:\DATA\Brioude_30092021\05-Traitements'
odir = r'C:\DATA\Brioude_30092021\05-Traitements\processing'
i = os.path.join(idir, 'C3_ground_thin_1m.sbf')
sbf_out = os.path.join(odir, 'C3_ground_thin_1m_clusters.sbf')

pc, sf, config = cc.read_sbf(i)

eps = 2
min_samples = 5

db = DBSCAN(eps=eps, min_samples=min_samples).fit(pc)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

cc.write_sbf(sbf_out, pc, labels.reshape(-1, 1))

print(f'{sbf_out} stored')