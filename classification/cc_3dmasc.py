# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 18:12:03 2022

@author: Baptiste Feldmann / Mathilde Letard
"""
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import sklearn
from sklearn import metrics

from ..tools import cc

#  definition of classes names and label values (here, internal conventions of the Rennes lidar platform)
classes = {2: 'Ground', 3: 'Low_veg', 4: 'Interm_veg', 5: 'High_veg.', 6: 'Building', 9: 'Water',
           11: 'Artificial_ground', 13: 'Power_Line', 14: 'Surf_zone', 15: 'Water_Column', 16: 'Bathymetry',
           18: 'Sandy_seabed', 19: 'Rocky_seabed', 23: 'Bare_ground', 24: 'Pebble', 25: 'Rock', 28: 'Car',
           29: 'Swimming_pools'}
# classes = {0: 'Ground', 1: 'Low_veg', 2: 'Interm_veg', 3: 'High_veg.', 4: 'Building', 5: 'Water',
#            6: 'Artificial_ground', 7: 'Power_Line', 8: 'Surf_zone', 9: 'Water_Column', 10: 'Bathymetry',
#            11: 'Sandy_seabed', 19: 'Rocky_seabed', 23: 'Bare_ground', 24: 'Pebble', 25: 'Rock', 28: 'Car',
#            29: 'Swimming_pools'}


def load_sbf_features(sbf_filepath, params_filepath, labels=False, coords=False):
    """
    Loading computed features after using 3DMASC plugin

    Parameters
    ---------
    sbf_filepath : str, absolute path to core-points file
    params_filepath : str, parameters file for 3DMASC
    labels : bool (default=False), in case of training model you need the labels
    coords : bool (default=False), if you want to get the coordinates too

    Returns
    --------
    data : dict,
         'features' : numpy.array of computed features
         'names' : list of str, name of each column feature
         'labels' : list of int, class labels
         'coords' : numpy.array of point coordinates
    """
    convention = {"NumberOfReturns": "Number Of Returns",
                  "ReturnNumber": "Return Number"}
    sf_dict = cc.get_name_index_dict(cc.read_sbf_header(sbf_filepath))
    for sfn in sf_dict.keys():
        sfn = sfn.replace(' ', '_')
    #features = np.loadtxt(params_filepath[0:-4]+"_feature_sources.txt",str)
    f = open(params_filepath[0:-4]+"_feature_sources.txt", "r")
    features = f.readlines()
    # if len(features.shape) == 0:
    #     features = [features.tolist()]
    sf_to_load = []
    loaded_sf_names = []
    for i in features[1:]:
        feature_name = i.split(sep=':')[1]
        feature_name = feature_name.split("\n")[0]
        base = feature_name.split('_')[0]
        if feature_name in convention.keys():
            sf_to_load.append(sf_dict[convention[feature_name]])
            loaded_sf_names.append(convention[feature_name])
        elif base in convention.keys():
            sf_to_load.append(sf_dict[feature_name.replace(base, convention[base])])
            loaded_sf_names.append(feature_name.replace(base, convention[base]))
        elif "kNN" in feature_name:
            sf_to_load.append(sf_dict['"'+feature_name+'"'])
            loaded_sf_names.append(feature_name)
        else:
            sf_to_load.append(sf_dict[feature_name])
            loaded_sf_names.append(feature_name)
    pc, sf, config = cc.read_sbf(sbf_filepath)
    data = {"features": sf[:, sf_to_load], "names": np.array(loaded_sf_names)}
    if labels:
        data["labels"] = sf[:, sf_dict['Classification']]
    if coords:
        data['coords'] = pc
    return data


def feature_clean(features):
    """
    Function to clean the features (no normalization, juste NaN and Inf values cleaning)

    Parameters
    ----------
    features : numpy array
        input features dataset.

    Returns
    -------
    dataset : numpy array
        a clean dataset.
    """
    for i in range(0, len(features[0, :])):
        col = features[:, i]
        if all(np.isnan(col)):
            newcol = np.array([-9999] * len(col))
        else:
            maxi = max(col[np.isfinite(col)])
            newcol = col
            newcol[np.isinf(col)] = maxi
            newcol[np.isnan(newcol)] = -9999
        features[:, i] = newcol
    return features


def train(trads, model=0):
    """
    Function to train a random forest model for point cloud features classification

    Parameters
    ----------
    trads : dictionary of numpy arrays
        training features dictionary.
    model : int (0 or 1)
        type of model. 0 = scikit-learn random forest, 1 = OpenCV random forest

    Returns
    -------
    model : sklearn RandomForestClassifier or OpenCV RTrees
        trained random forest model ready for use.
    """
    if model == 0:
        classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=150,criterion='gini', max_features="sqrt",
                                                             max_depth=None, oob_score=True, n_jobs=-1, verbose=0)
        classifier.fit(feature_clean(trads['features']), trads['labels'])
    elif model == 1:
        labels = np.array(trads['labels']).astype('int32')
        classifier = cv2.ml.RTrees_create()
        classifier.setMaxDepth(25)
        classifier.setActiveVarCount(0)
        classifier.setCalculateVarImportance(True)
        classifier.setMinSampleCount(1)
        term_type, n_trees, epsilon = cv2.TERM_CRITERIA_MAX_ITER, 150, sys.float_info.epsilon
        classifier.setTermCriteria((term_type, n_trees, epsilon))
        train_data = cv2.ml.TrainData_create(samples=np.array(trads['features']).astype('float32'),
                                             layout=cv2.ml.ROW_SAMPLE, responses=labels)
        classifier.train(trainData=train_data)
    else:
        print('Invalid model type')
        return
    return classifier


def test(testds, classifier, model=0):
    """
    Function to test the random forest model obtained on the test dataset and compute classification metrics.

    Parameters
    ----------
    testds : dict of numpy arrays
        test features dictionary.
    classifier : sklearn RandomForestClassifier or OpenCV RTrees
        trained random forest model ready for use.
    model : int (0 or 1)
        type of the model. 0 = scikit-learn random forest, 1 = OpenCV random forest

    Returns
    -------
    labels_pred : numpy array
        labels predicted by the model for each set of features.
    confid_pred : numpy array
        prediction confidence for each set of features.
    feat_imptce : numpy array
        importance of each feature (as computed in sklearn's RandomForestClassifier).
    oa : float
        overall accuracy of the classifier
    fs : float
        F1-score of the classifier averaged on all classes

    """
    labels = testds['labels']
    if model == 0:
        feat_imptce = classifier.feature_importances_
        labels_pred = classifier.predict(feature_clean(testds['features']))
        conf = classifier.predict_proba(feature_clean(testds['features']))
    elif model == 1:
        feat_imptce = classifier.getVarImportance()
        _ret, responses = classifier.predict(testds['features'].astype('float32'))
        votes = classifier.getVotes(testds['features'].astype('float32'), 0)
        conf = votes[1:, :]
        conf = np.max(conf, axis=-1)/150
        labels_pred = responses
    else:
        print('Invalid model type')
        return
    oa = metrics.accuracy_score(labels, labels_pred)
    fs = metrics.f1_score(labels, labels_pred, average='macro')
    return labels_pred, conf, feat_imptce, oa, fs


def get_acc_expe(trads, testds, plot=True, save=False, model=0):
    """
    Function to train a random forest model for point cloud features classification

    Parameters
    ----------
    trads : dictionary of numpy arrays
        training features dictionary.
    testds : dict of numpy arrays
        test features dictionary.
    save : bool
        defines if plot must be saved.
    plot : bool
        defines if plot must be opened.
    model : int (0 or 1)
        type of model. 0 = scikit-learn random forest, 1 = OpenCV random forest

    Returns
    -------
    accuracy : float    Overall Accuracy of classifier
    fscore : float      F1-score (averaged on all classes)
    numpy.mean(confid_pred) : float     Mean prediction confidence
    recall : float      Recall (averaged on all classes)
    precision : float      Recall (averaged on all classes)
    uas : numpy.array(float)    User's accuracies (per class)
    pas : numpy.array(float)    Producer's accuracies (per class)
    fscores : numpy.array(float)     F1-score per class
    confc : numpy.array(float)     Mean prediction confidence per class
    recalls : numpy.array(float)     Recall per class
    precisions : numpy.array(float)     Precision per class
    labels : numpy.array(float)     labels
    feat_imptce : numpy.array(float)     feature importance values
    classifier : skleanrn RandomForestClassifier or OpenCV RTrees   classifier
    labels_pred : np.array(int)     model predictions

    """
    classifier = train(trads, model)
    labels_pred, confid_pred, feat_imptce, oa, fs = test(testds, classifier, model)
    test_labels = testds['labels']
    accuracy = metrics.accuracy_score(test_labels, labels_pred)
    precision = metrics.precision_score(test_labels, labels_pred, average='macro')
    recall = metrics.recall_score(test_labels, labels_pred, average='macro')
    fscore = metrics.f1_score(test_labels, labels_pred, average='macro')
    mat_pa = metrics.confusion_matrix(test_labels, labels_pred, normalize='true')
    mat_ua = metrics.confusion_matrix(test_labels, labels_pred, normalize='pred')
    confmat = metrics.confusion_matrix(test_labels, labels_pred, normalize=None)
    labels = np.unique(test_labels)
    figure = go.Figure(data=go.Heatmap(
                   z=confmat,
                   x=[classes[l] for l in labels],
                   y=[classes[l] for l in labels],
                   text=confmat,
                   texttemplate="%{text}",
                   textfont={"size": 20},
                   colorscale=[(0, "rgb(250,250,250)"), (0.25, 'darkseagreen'), (1.0, "seagreen")],
                   hoverongaps=False))
    figure.update_layout(
        title_text="Confusion matrix",
        font_size=18,
    )
    if save:
        figure.write_html('confusion_matrix.html', auto_open=False)
    if plot:
        figure.write_html('confusion_matrix.html', auto_open=True)
    uas = []
    pas = []
    for i in range(mat_pa.shape[0]):
        pas.append(mat_pa[i, i])
        uas.append(mat_ua[i, i])
    labels = np.unique(test_labels)
    confc = []
    for l in labels:
        confc.append(np.mean(confid_pred[np.where((labels_pred == l))[0]]))
    precisions = metrics.precision_score(test_labels, labels_pred, average=None)
    recalls = metrics.recall_score(test_labels, labels_pred, average=None)
    fscores = metrics.f1_score(test_labels, labels_pred, average=None)
    return accuracy, fscore, np.mean(confid_pred), recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imptce, classifier, labels_pred


def plot_feat_imp(feat_imp, trads, save=False, plot=True):
    """
    Function to get a graph plot of the RF model's feature importances

    Parameters
    ----------
    feat_imp : numpy.array()
        array containing feature importances values.
    trads : dictionary of numpy arrays
        training features dictionary.
    save : bool
        defines if plot must be saved.
    plot : bool
        defines if plot must be opened.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='Feature Importances', x=trads['names'], y=feat_imp, marker_color='navy'),
    )
    fig.update_layout(
        title_text="<b>Feature importances<b>"
    )
    fig.update_xaxes(tickangle=45, showgrid=True, type='category', title_text='<b>Feature<b>')
    fig.update_yaxes(title_text="<b>RF Importance</b>")
    if plot:
        fig.write_html( 'feature_importances.html', auto_open=True)
    if save:
        fig.write_image('feature_importances.jpg', scale=3)


def plot_corr_mat(trads, plot=True, save=False):
    """
    Function to visualize correlation between features.

    Parameters
    ----------
    trads : dictionary of numpy arrays
        training features dictionary.
    save : bool
        defines if plot must be saved.
    plot : bool
        defines if plot must be opened.
    """
    feats_df = pd.DataFrame(trads['features'], columns=trads['names'])
    corr_mat = feats_df.corr()
    figure = go.Figure(data=go.Heatmap(
                   z=corr_mat,
                   x=trads['names'],
                   y=trads['names'],
                   text=np.round(corr_mat*100, 0),
                   texttemplate="%{text}",
                   textfont={"size": 20},
                   colorscale='magma',
                   hoverongaps=False))
    figure.update_layout(
        title_text="Correlation matrix",
    )
    if save:
        figure.write_html('correlation_matrix.html', auto_open=False)
    if plot:
        figure.write_html('correlation_matrix.html', auto_open=True)

    return corr_mat


def get_shap_expl(classifier, testds, save=True):
    """
    Function to get the shap summary plot of a random forest classifier trained on the given dataset

    Parameters
    ----------
    classifier : scikit-learn RandomForestClassifier
        trained classifier.
    testds : dict,
         'features' : numpy.array of computed features
         'names' : list of str, name of each column feature
         'labels' : list of int, class labels
        training dataset.
    save : bool
        whether to save the resulting plot.
    """
    labels = np.unique(testds['labels'])
    cn = []
    for l in labels:
        cn.append(classes[int(l)])
    explainer = shap.TreeExplainer(classifier)
    fig = plt.figure()
    fig.set_size_inches(32, 18)
    shap_values = explainer.shap_values(testds['features'], approximate=True)
    shap.summary_plot(shap_values, testds['features'], feature_names=testds['names'], class_names=cn,
                      max_display=testds['features'].shape[1], plot_type="bar")
    if save:
        plt.savefig('SHAP_explainer.jpg', bbox_inches='tight')
    return shap_values


def classif_errors_confidence(pred, true, confid_pred):
    idx_err = np.where((pred != true))[0]
    err_confid = confid_pred[idx_err]
    err_stats = {'Mean_confidence': np.mean(err_confid), 'Median_confidence': np.median(err_confid),
                 'Min_confidence': np.min(err_confid), 'Max_confidence': np.max(err_confid),
                 'Std_confidence': np.std(err_confid)}
    return err_stats


def apply_confidence_threshold(pred, true, confid_pred, threshold):
    idx_ok = np.where(confid_pred >= threshold)[0]
    accuracy = metrics.accuracy_score(true[idx_ok], pred[idx_ok])
    return accuracy


def confidence_filtering_report(pred, true, confid_pred):
    thresholded_acc = {0.5: '', 0.6: '', 0.7: '', 0.8: '', 0.9: '', 0.95: ''}
    percent = {0.5: '', 0.6: '', 0.7: '', 0.8: '', 0.9: '', 0.95: ''}
    for t in thresholded_acc.keys():
        idx_ok = np.where(confid_pred >= t)[0]
        accuracy = metrics.accuracy_score(true[idx_ok], pred[idx_ok])
        thresholded_acc[t] = accuracy
        percent[t] = len(idx_ok) / len(pred)
    return thresholded_acc, percent
