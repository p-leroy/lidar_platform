.. _3dmasc:

======
3DMASC
======

3DMASC is a plugin for CloudCompare developped by the LiDAR platform in Rennes. The general documentation is `here <https://lidar.univ-rennes.fr>`_. In the following section we present the work we did in Python to help you run 3DMASC or to optimize a classifier in an advanced workflow.

Remember that plugin can be called in command line and that there are 4 options for that:

- 3DMASC_CLASSIFY: the main option for calling 3DMASC in command line, it is mandatory.
- ONLY_FEATURES: do not train the classifier, just compute the features (used when the training is to be done with Python).
- KEEP_ATTRIBUTES: at the end of the computation, the features will be stored.
- SKIP_FEATURES: do not compute the features (used when the features have already been computed).

The 3DMASC plugin can be called from Python using a helper function of ``lidar_platform.cc`` which builds the command line for you and run it
using ``subprocess``.

.. code-block::

    from lidar_platform import cc

Basic actions
=============

Classify
--------

Once you have build your classifier with CloudCompare, it is possible to apply it to another set of data using the library.

The following Python code:

.. code-block::

    clouds = (pc1, pc2, core)  # pc1, pc2 and core are full paths to clouds
    out = cc.q3dmasc(clouds, classifier)

is equivalent to this command:

.. code-block:: shell

    CloudCompare.exe -SILENT -NO_TIMESTAMP -C_EXPORT_FMT sbf -o -GLOBAL_SHIFT AUTO pc1 -o -GLOBAL_SHIFT AUTO pc2 -o -GLOBAL_SHIFT AUTO core -3DMASC_CLASSIFY classifier.txt C3HD=2 C2HD=1 CORE=3

**Be carefull to the order of the clouds in the classifier file**, it may differ from the order in the parameter file. You should take care of the coherence of the order of the cloud list, either using Python or the command line when calling 3DMASC.

Compute features
----------------

If you want to only compute the features on a specific point cloud, the following Python code:

.. code-block::

    clouds = (pc1, pc2, core)  # pc1, pc2 and core are full paths to clouds
    out = cc.q3dmasc(clouds, parameters, only_features=True, verbose=True)

is equivalent to this command:

.. code-block:: shell

    CloudCompare.exe -SILENT -NO_TIMESTAMP -C_EXPORT_FMT sbf -o -GLOBAL_SHIFT AUTO pc1 -o -GLOBAL_SHIFT AUTO pc2 -o -GLOBAL_SHIFT AUTO core -3DMASC_CLASSIFY -ONLY_FEATURES 3dmasc_parameters.txt C2HD=1 C3HD=2 CORE=3

If your parameter file is named *3dmasc_parameters.txt*, a file with the name *3dmasc_parameters_feature_sources.txt* is
created by CloudCompare, note that this file is used by other functions of the library.

Import a point cloud with features
----------------------------------

Once the features have been calculated, and if you have chosen to save the results in sbf you can easily import the cloud with its features using the ``lidar_platform.sbf`` module. Features are stored as scalar fields by CloudCompare.

.. code-block::

    from lidar_platform import sbf
    sbf_data = sbf.read(sbf_file)  # sbf_data is an object of type SbfData
    xyz = sbf.pc  # coordinates of the points
    scalar_fields = sbf.sf  # scalar fields as a NumPy array
    scalar_fields_names = sbf.sf_names  # scalar fields names

When 3DMASC computes one feature, it may also compute intermediate scalar fields which are used to compute the feature itself. So all scalar fields are not features. For instance, when computing *DimZ_C2HD_MODE_MINUS_DimZ_C3HD_MODE@1.5*, 3DMASC needs to compute *DimZ_C3HD_MODE@1.5* which is also stored in the scalar fields but not used as a feature unless you specified it explicitly in the 3DMASC parameter file.

This is where the file *3dmasc_parameters_feature_sources.txt* created during the feature calculation is important, it let you know exactly which features were used to train the classifier. This file is used by the function ``classification.cc_3dmasc.load_sbf_features`` to load only the features related to the classifier.

Once the features have been calculated, it is possible to load them in Python, jointly with the classifier, to classify data directly in Python.

Classify in Python with OpenCV
------------------------------

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

Advanced workflow
=================

For more advanced classification problems, a possibility is to use our Python scripts, which allow you to dive deeper into the classifier details.

To replicate classifier optimization as described in the original `3DMASC paper <https://www.sciencedirect.com/science/article/pii/S0924271623003337?via%3Dihub>`_, you can follow the steps below, which require having the lidar_platform Python package installed and working.

First steps in Python with 3DMASC
---------------------------------

1. Create your parameter file as explained above.
2. Compute the features on your training point cloud using the command line. Using the command line interface instead of the GUI will automatically generate a file with a name ending by *_feature_sources.txt*. **This file is mandatory to use the workflow**. Save the resulting point cloud (with the features) in the SBF format. Repeat this operation on your test point cloud and your core points point cloud.
3. Import lidar_platform.classification
4. Load the different point clouds in Python using classification.cc_3dmasc.load_sbf_features()

Training and evaluating a classifier in Python
----------------------------------------------

* use classification.cc_3dmasc.train() to train a random forest model
* use classification.cc_3dmasc.get_acc_expe() to train a random forest model and obtain multiple metrics quantifying its performance
* use classification.cc_3dmasc.test() to evaluate your model on a test dataset

In this module, you can also find functions to visualize the random forest feature importance, heatmaps of inter-feature linear correlation, or a SHAP summary plot.

Classifier optimization in Python
---------------------------------

The optimization procedure detailed in the 3DMASC paper consists of iteratively pruning a set of features and scales using embedded random forest metrics and using the variation in classification performance to identify an optimal set of predictors. In the Python library, the functions classification.feature_selection.rf_ft_selection() and classification.feature_selection.get_best_rf_select_iter() will enable you to do the same for your classification workflows.

In lidar_platform.classification.feature_selection you will also find multiple scripts to perform other operations such as selecting a given number of uncorrelated features or scales. You can also directly select a set of predictors made of at most n features and m scales using get_n_optimal_sc_ft.

Shifting between GUI and Python for classifier training and application
-----------------------------------------------------------------------

In some cases, it can be interesting to shift from the CloudCompare GUI to Python or vice-versa. In particular, when using very large datasets, it can be useful to perform classifier training in Python rather than in the GUI. In any case, a classifier that is trained in Python can be used in CloudCompare and vice-versa.

Namely, to load in Python a classifier saved from the q3DMASC plugin, use:

.. code-block::

    import cv2
    cl = cv2.ml.RTrees_create()
    cl = cl.load('your_classifier.yaml')

When using Python, you can also save a trained OpenCV random forest model and use it in the CloudCompare plugin (using the save function of ``cv2.ml.RTrees``).

Since you cannot open a .yaml file directly in the 3DMASC classify tool, you will have to indicate the path to the classifier at the top of the parameters file (when doing all the process in the plugin, after training, the classifier is saved and the parameters file is automatically updated with the path of the .yaml model, which enabled the plugin to find it. Here, it is necessary to do it manually). For that, simply add these lines at the top of the parameters file, and then use 3DMASC classify as usual.

.. code-block::

    # 3DMASC classifier file
    classifier: classifier.yaml

(replace bolded text with adapted path and file name).