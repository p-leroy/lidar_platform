# Baptiste Feldman
# Paul Leroy

import glob
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from lidar_platform.classification import cc_3dmasc
from lidar_platform import tools

workspace = r'G:\RENNES1\BaptisteFeldmann\Python\training\Loire\juin2019\classif_C3_withSS\dalles' + '//'
features_file = "Loire_20190529_C3_params_v3.txt"
query0_CC = ['standard','SBF','Loire']

list_pcx = [os.path.split(i)[1] for i in glob.glob(workspace + "PCX_*.laz")]
print("%i files found !" % len(list_pcx))
print("================================")

# ---Compute features---#
deb = time.time()
for i in list_pcx:
    print(i + " " + str(list_pcx.index(i) + 1) + "/" + str(len(list_pcx)))
    if not os.path.exists(workspace + "features/" + i[0:-4] + "_features.sbf"):
        cc_3dmasc.compute_features(workspace, i, query0_CC, workspace + features_file)
    print("================================")
print("Time duration: %.1f sec" % (time.time()-deb))

list_sbf = glob.glob(workspace + "features/*_features.sbf")
query = tools.cloudcompare.open_file(query0_CC, list_sbf)
tools.cloudcompare.merge_clouds(query)
tools.cloudcompare.last_file(workspace + "features/*_MERGED_*.sbf",
                          "PCX_all_features.sbf")
tools.cloudcompare.last_file(workspace + "features/*_MERGED_*.sbf.data",
                          "PCX_all_features.sbf.data")

print("Compute features time duration: %.1f sec" % (time.time()-deb))

# ---Initialization---#
dictio = cc_3dmasc.load_features(workspace + "features/PCX_all_features.sbf", workspace + features_file, True)
# features normalization :
# NaN are replaced by -1 and for each feature min=0 and max=1
data = MinMaxScaler((0,1)).fit_transform(dictio['features'])
data = np.nan_to_num(data, nan=-1)

names = dictio['names']
labels = dictio['labels']

model = RandomForestClassifier(n_estimators=500, criterion='gini', max_features="auto",
                               max_depth=None, oob_score=True, n_jobs=50, verbose=1)

# ---Cross Validation---#
NbFold = 10
deb = time.time()
skf = StratifiedKFold(n_splits=NbFold, shuffle=True, random_state=42)
kappa, OA, feat_import = cc_3dmasc.cross_validation(model, skf, data, labels)
print("CV time duration: %.1f sec" %(time.time()-deb))
print(kappa, OA, feat_import, sep="\n")

outFile = open(workspace + "test_CrossValidation_3.pkl", 'wb')
pickle.dump(
    {"kappa": kappa, "OA": OA, "feat_import": feat_import},
    outFile)
outFile.close()

# ---Training---#
model.fit(data, labels)
outFile = open(workspace + "Loire_Rtemus2019_C3_HR_model_v3.pkl", "wb")
pickle.dump(model, outFile)
outFile.close()
