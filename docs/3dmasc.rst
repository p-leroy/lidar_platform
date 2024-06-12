.. _3dmasc:

======
3DMASC
======

3DMASC is a plugin for CloudCompare developped by the LiDAR platform in Rennes. Its documentation is `here <https://lidar.univ-rennes.fr>`_.

The plugin can be called in command line and there are 4 options for that:

- 3DMASC_CLASSIFY: the main option for calling 3DMASC in command line, it is mandatory.
- ONLY_FEATURES: do not train the classifier, just compute the features (used when the training is to be done with Python).
- KEEP_ATTRIBUTES: at the end of the computation, the features will be stored.
- SKIP_FEATURES: do not compute the features (used when the features have already been computed).

The 3DMASC plugin can be called from Python using a helper function of ``lidar_platform.cc`` which builds the command line for you and run it
using ``subprocess``.

.. code-block::

    from lidar_platform import cc

Classify
========

Once you have build your classifier with CloudCompare, it is possible to apply it to another set of data using the library.

The following Python code:

.. code-block::

    clouds = (pc1, pc2, core)
    out = cc.q3dmasc(clouds, classifier)

is equivalent to this command:

.. code-block:: shell

    CloudCompare.exe -SILENT -NO_TIMESTAMP -C_EXPORT_FMT sbf -o -GLOBAL_SHIFT AUTO C3_HD.laz -o -GLOBAL_SHIFT AUTO C2_HD.laz -o -GLOBAL_SHIFT AUTO core.laz -3DMASC_CLASSIFY classifier.txt C3HD=2 C2HD=1 CORE=3

**Be carefull to the order of the clouds in the classifier file**, it may differ from the order in the parameter file. You should take care of the coherence of the order of the cloud list, either using Python or the command line when calling 3DMASC.

Compute features
================

If you want to only compute the features on a specific point cloud, the following Python code:

.. code-block::

    clouds = (pc1, pc2, core)
    out = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True)

is equivalent to this command:

.. code-block:: shell

    CloudCompare.exe -SILENT -NO_TIMESTAMP -C_EXPORT_FMT sbf -o -GLOBAL_SHIFT AUTO C2_HD.laz -o -GLOBAL_SHIFT AUTO C3_HD.laz -o -GLOBAL_SHIFT AUTO core.laz -3DMASC_CLASSIFY -ONLY_FEATURES 3dmasc_parameters.txt C2HD=1 C3HD=2 CORE=3

If your parameter file is named *3dmasc_parameters.txt*, a file with the name *3dmasc_parameters_feature_sources.txt* is
created by CloudCompare, note that this file is used by other functions of the library.

Import a point cloud with features
==================================

Once the features have been calculated, and if you have chosen to save the results in sbf you can easily import the cloud with its features using the ``lidar_platform.sbf`` module. Features are stored as scalar fields by CloudCompare.

.. code-block::

    from lidar_platform import sbf
    sbf_data = sbf.read(sbf_file)  # sbf_data is an object of type SbfData
    xyz = sbf.pc  # coordinates of the points
    scalar_fields = sbf.sf  # scalar fields as a NumPy array
    scalar_fields_names = sbf.sf_names  # scalar fields names

When 3DMASC computes one feature, it may also compute intermediate scalar fields which are used to compute the feature itslef. So all scalar fields are not features. For instance, when computing *DimZ_C2HD_MODE_MINUS_DimZ_C3HD_MODE@1.5*, 3DMASC needs to compute *DimZ_C3HD_MODE@1.5* which is also stored in the scalar fields but not used as a feature unless you specified it explicitly in the 3DMASC parameter file.

This is where the file *3dmasc_parameters_feature_sources.txt* created during the feature calculation is important, it let you know exactly which features were used to train the classifier. This file is used by the function ``classification.cc_3dmasc.load_sbf_features`` to load only the features related to the classifier.

Once the features have been calculated, it is possible to load them in Python, jointly with the classifier, to classify data directly in Python.

Classify in Python with OpenCV
==============================

When saving a classifier, CloudCompare saves two files: a txt file which contains a reference to the classifier itself which is in YAML (this one can be loaded in Python with the cv2 module). The yaml file can be loaded with ``cv2``.

To run the following lines of code, you will need to compute the features on your point cloud as presented above. A file *3dmasc_parameters_feature_sources.txt* is mandatory, it should be in the same directory as *3dmasc_parameters.txt*.

.. code-block::

    import cv2
    from lidar_platfor import classification, sbf

    # Load features from sbf
    features_data = classification.cc_3dmasc.load_sbf_features(
        point_cloud_WITH_FEATURES.sbf,
        3dmasc_parameters.txt)
    x_test = features_data['features']

    # Load a classifier and apply it
    cls = cv2.ml.RTrees_load(opencv_classifier)  # load
    _, y_pred = cls.predict(x_test, flags=cv2.ml.DTREES_PREDICT_MAX_VOTE)  # apply

    # Read the data from the sbf file
    sbf_data = sbf.read(test_with_features)

    # Add an extra scalar field called class_python to the existing ones
    sbf_data.add_sf("class_python", y_pred)

    # Save the point cloud in sbf format
    sbf.write(output_file, sbf_data.pc, sbf_data.sf, sbf_data.config)
