import os

import cv2

from lidar_platform import cc, classification, sbf

dir_ = r'C:\DATA\test_3dmasc'

pc1 = os.path.join(dir_, "C2_HD.laz")
pc2 = os.path.join(dir_, "C3_HD.laz")
core = os.path.join(dir_, "core.laz")
test = os.path.join(dir_, "test.laz")
test_with_features = r'C:\DATA\test_3dmasc\test_WITH_FEATURES.sbf'
parameters = os.path.join(dir_, "3dmasc_parameters.txt")  # 3DMASC parameter file
# when saving a classifier, CloudCompare save two files: a file in TXT which contains a reference to the classifier
# itself in YAML (this one can be loaded in Python with the cv2 module)
classifier = os.path.join(dir_, "classifier.txt")  # a classifier build using CloudCompare in GUI mode
opencv_classifier = os.path.join(dir_, "classifier.yaml")  # a classifier build using CloudCompare in GUI mode

#%% Compute the features
# outputs
#   core_WITH_FEATURES.sbf + core_WITH_FEATURES.sbf.data
#   3dmasc_parameters_feature_sources.txt
clouds = (pc1, pc2, test)
out = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True)

#%% Compute features + classify
# outputs
#   core_CLASSIFIED.sbf + core_CLASSIFIED.sbf.data
#
clouds = (pc2, pc1, core)
out = cc.q3dmasc(clouds, classifier, verbose=True)

#%% Read the features
# WARNING: the file 3dmasc_parameters_feature_sources.txt associated with 3dmasc_parameters.txt shall exist.
features_data = classification.cc_3dmasc.load_sbf_features(
    r'C:\DATA\test_3dmasc\test_WITH_FEATURES.sbf',
    r'C:\DATA\test_3dmasc\3dmasc_parameters.txt')
x_test = features_data['features']

#%% Load an OpenCV classifier
cls = cv2.ml.RTrees_load(opencv_classifier)

#%% Apply the classifier
_, y_pred = cls.predict(x_test, flags=cv2.ml.DTREES_PREDICT_MAX_VOTE)

#%% Add the predicted class to the scalar fields
sbf_data = sbf.read(test_with_features)
sbf_data.add_sf("class_python", y_pred)

#%% Save the point cloud in sbf format
sbf.write(r'C:\DATA\test_3dmasc\test_PYTHON.sbf', sbf_data.pc, sbf_data.sf, sbf_data.config)
