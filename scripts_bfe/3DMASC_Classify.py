# Baptiste Feldman
# Paul Leroy

import glob
import os
import pickle
import time

from lidar_platform.classification import cc_3dmasc

workspace = r'G:\RENNES1\Loire_totale_automne2019\Loire_Briare-Sully-sur-Loire\05-Traitements\C3\classification\bathy\haute_resolution' + '//'
list_ = glob.glob(workspace + "PCX_*00.laz")
names = [os.path.split(i)[1] for i in list_]
print("%i files found !" % (len(names)))

classifier = {
    "path": r'G:\RENNES1\BaptisteFeldmann\Python\training\Loire\juin2019\classif_C3_withSS'+'//',
    "features_file": "Loire_20190529_C3_params_v3.txt",
    "model": "Loire_Rtemus2019_C3_HR_model_v3.pkl"}
queryCC_param = ['standard','SBF','Loire45-4']

#computeFeatures(workspace,"PCX_660000_6739000.laz",queryCC_param,classifier["path"]+classifier["features_file"])


deb = time.time()
for i in names:
    print(i + " " + str(names.index(i) + 1) + "/" + str(len(names)))
    if not os.path.exists(workspace+"features/" + i[0:-4] + "_features.sbf"):
        cc_3dmasc.compute_features(workspace, i, queryCC_param, classifier["path"] + classifier["features_file"])
    print("================================")
print("Time duration: %.1f sec" % (time.time() - deb))

infile = open(classifier["path"]+classifier["model"],"rb")
tree = pickle.load(infile)
infile.close()
tree.verbose = 1
tree.n_jobs = 50

#classify(workspace,"PCX_660000_6739000.laz",tree,classifier["path"]+classifier["features_file"])

for i in names:
    print(i)
    if not os.path.exists(workspace + i[0:-4] + "_class.laz"):
        classification.cc_3dmasc.classify(workspace, i, tree, classifier["path"] + classifier["features_file"])

#Parallel(n_jobs=10,verbose=2)(delayed(classify)(workspace,i,tree,classifier["path"]+classifier["features_file"]) for i in liste_noms)
#=============================#




