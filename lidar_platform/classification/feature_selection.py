# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:33:50 2022

@author: Mathilde Letard
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn import feature_selection

from . import cc_3dmasc
from .cc_3dmasc import get_acc_expe, feature_clean


def get_scales_feats(ds):
    """
    Get the scales and features present in the dataset read by cc_3dmasc.load_sbf_features().

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names obtained using load_sbf_features().

    Returns
    -------
    numpy array : (ns*nf) x 1
        list containing the scale of each descriptor.
    numpy array : nf x 1
        list containing the feature name of each descriptor.
    numpy array : (ns*nf) x 1
        list containing the complete name of each descriptor.
    """
    scales = []
    names = []
    for i in ds['names']:
        name = i
        if 'kNN' not in i:
            split = i.split(sep='@')
            if len(split) > 1:
                scales += [float(split[1])]
            else:
                scales += [0]
            names += [name.split('@')[0]]
        else:
            scales += [0]
            names += [name]
    return np.array(scales), np.array(names), ds['names']


def nan_percentage(ds):
    """
    Get the percentage of NaN values for each feature. This can be useful to better understand why a feature at a
    given scale is not contributing, or to identify relevant minimal scales to use.
    (reminder: 3DMASC outputs NaN for points for which the feature was impossible to compute - for ex. due to
    no neighbors in the specified sphere scale).

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().

    Returns
    -------
    dictio_ft : dictionary
        dictionary containing the name of each feature and the associated percentage of NaN.
    """
    scales, names, ds_names = get_scales_feats(ds)
    features = ds['features']
    feats, indices = np.unique(names, return_index=True)
    sort_ind_arg = np.argsort(indices)
    sorted_indices = indices[sort_ind_arg]
    sorted_feats = feats[sort_ind_arg]
    percent = []
    ok = False
    for f in range(len(sorted_feats)):
        if not ok:
            try:
                feat_data = features[:, sorted_indices[f]:sorted_indices[f+1]]
                scales_feat = scales[sorted_indices[f]:sorted_indices[f+1]]
            except IndexError:
                feat_data = features[:, sorted_indices[f]:]
                scales_feat = scales[sorted_indices[f]:]
            if len(scales_feat) > 2:
                ok = True
            if ok:
                for j in range(feat_data.shape[1]):
                    percent.append(len(np.where(np.isnan(feat_data[:, j]))[0])/feat_data.shape[0])

                dictio_ft = {'Scales': scales_feat, 'Percentage': percent}
                return dictio_ft


def info_score(ds):
    """
    Get the mutual information score of each feature (computed with respect to the labels to predict).
    This metric is used in the classifier optimization procedure.

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().

    Returns
    -------
    dictio_ft : dictionary
        contains the name of each feature and the associated score value.
    """
    ds_cleaned = feature_clean(ds['features'])
    mi = sklearn.feature_selection.mutual_info_classif(ds_cleaned, ds['labels'])
    dictio_ft = {'Features': ds['names'], 'MutualInfo': mi}
    return dictio_ft


def inter_ft_corr_filter(features_set, features_score, threshold):
    """
    Prune a set of predictors by keeping only the most informative elements among correlated pairs.
    First, linear correlation between all provided features at all provided scales is computed.
    Then, when the correlation between two predictors exceeds a given threshold, only the one with the highest
    mutual information score is kept.

    Parameters
    ----------
    features_set : numpy array (n_points x n_predictors)
        array containing the value of each predictor evaluated for each point (e.g., "features" field of the
        dictionary obtained with load_sbf_features()).
    features_score : numpy array (n_predictors x 1)
        array containing the information score of each predictor (obtained with info_score()).
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    select : list(int)
        indices of the selected features in the original features_set array (column indices).
    """
    feats_df = pd.DataFrame(features_set)
    corr_mat = feats_df.corr().to_numpy()
    half = np.abs(corr_mat)
    for i in range(corr_mat.shape[0]):
        half[i, i] = 0
    select = np.arange(0, features_set.shape[1], 1)
    cor = np.where((half >= threshold))
    todel = []
    if len(cor[0]) > 0:
        for f in range(len(cor[0])):
            comp = [cor[0][f], cor[1][f]]
            todel.append(comp[np.argmin(np.array(features_score)[comp])])
    select = np.delete(select, todel, axis=0)
    return select  # indices of selected features


def filter_corr_with_selected_ft(all_ft, candidate_ft, selected_ft, threshold):
    """
    Check compatibility of features considering their linear correlation with an existing set of features.
    This function allows to evaluate whether a new predictor can be added to a set of previously selected features and
    scales without overcoming the maximum accepted inter-feature linear correlation coefficient.

    Parameters
    ----------
    all_ft : numpy array (n_points x n_predictors)
        array containing the value of each predictor for each point ("features" fields of the dict obtained with
        obtained using load_sbf_features()).
    candidate_ft : list(int)
        indices of the elements to consider for selection in the all_ft array (column index of the
        candidates to evaluate).
    selected_ft : numpy array (n_points x n_selected)
        array containing the features that are already selected.
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    valides : list(int)
        list of the indices of the selected features in the original ds['features'] array (obtained with
        load_sbf_features()).
    """
    feats_df = pd.DataFrame(all_ft[:, candidate_ft])
    corr_mat = feats_df.corr().to_numpy()
    half = np.abs(corr_mat)
    for i in range(corr_mat.shape[0]):
        half[i, ] = 0
    select = np.arange(0, len(selected_ft), 1)  # indices of already selected features
    cor = np.where((half >= threshold))  # indices of correlated features
    todel = []
    for s in select:
        todel += cor[1][np.where((cor[0] == s))[0].tolist()].tolist()  # features correlated to validated features set
        todel += cor[0][np.where((cor[1] == s))[0].tolist()].tolist()  # features correlated to validated features set
    invalides = select.tolist() + np.unique(np.array(todel)).tolist()  # indices of invalid features
    valides = np.delete(candidate_ft, invalides)
    return valides.tolist()


def get_n_uncorr_ft(ft_all, ft_select, ft_score, nf, threshold):
    """
    Iteratively complete an unfilled set of uncorrelated features.
    This function iteratively looks for additional features to select to reach nf uncorrelated features.

    Parameters
    ----------
    ft_all : numpy array (n_points x n_predictors)
        array containing the value of each predictor for each point ("features" field of the dict
        obtained with load_sbf_features()).
    ft_select : numpy array (n_selected)
        index in ft_all of each predictor that already passed the selection.
    ft_score : numpy array (n_predictors x 1)
        array containing the information score of each predictor.
    nf : int
        number of different features to select.
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    ft_select : list(int)
        list of the indices of the selected features in the original ds['features'] array (obtained with
        load_sbf_features()).
    """
    n_select = len(ft_select)
    argsort_sc = np.argsort(ft_score)
    possib = argsort_sc[-1 * nf:]
    nb_tested = nf
    while n_select < nf and possib.shape[0] > 0:
        possib = argsort_sc[:-1 * nb_tested]
        rem = nf - n_select
        test = ft_select + possib[-1 * rem:].tolist()  # indices of selected features and best features
        compatibles = filter_corr_with_selected_ft(ft_all, test, ft_select, threshold)  # all selected features
        valides_id = inter_ft_corr_filter(ft_all[:, compatibles], ft_score[compatibles], threshold)
        valides = np.array(compatibles)[valides_id]
        ft_select += valides.tolist()
        n_select = len(ft_select)
        nb_tested += rem
    return ft_select


def n_best_uncorr_ft(ds, nf, corr_threshold):
    """
    Select nf uncorrelated features depending on their mutual information score and linear correlation.

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names obtained with cc_3dmasc.load_sbf_features().
    nf : int
        number of different features to select.
    corr_threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    select : list(int)
        list of the indices of the selected features in the original ds['features'] array (obtained with
        cc_3dmasc.load_sbf_features()).
    """
    ft_score = info_score(ds)['MutualInfo']
    possib = np.argsort(ft_score)[-1 * nf:]
    select_id = inter_ft_corr_filter(ds['features'][:, possib], ft_score[possib], corr_threshold).tolist()
    select = np.array(possib)[select_id]
    if len(select.tolist()) != nf:
        select = get_n_uncorr_ft(ds['features'], select.tolist(), ft_score, nf, corr_threshold)
    return select


def n_best_uncorr_sc(ds, n_scales, corr_threshold):
    """
    Select ns uncorrelated scales depending on their mutual information score, linear correlation,
    and a voting process.
    For each investigated features, all available scales are investigated and pruned depending on their correlations.
    Then the ns most frequently retained scales among all features are kept as the final set of scales.

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().
    n_scales : int
        number of different scales to select.
    corr_threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    optim_ok : list(float)
        list of selected scales
    freq_optim : list(int)
        number of votes obtained by each selected scale.
    """
    scales, names, ds_names = get_scales_feats(ds)
    best_scales = []
    for f in np.unique(names):
        f_id = np.where((names == f))[0]
        search_ds = {'features': ds['features'][:, f_id], 'labels': ds['labels'], 'names': ds_names[f_id]}
        result = n_best_uncorr_ft(search_ds, n_scales, corr_threshold)
        best_scales += scales[result].tolist()
    sc, freq = np.unique(best_scales, return_counts=True)
    frequencysorted = freq[np.argsort(freq)]
    uniq_freq = np.sort(np.unique(frequencysorted))
    optim = []
    freq_optim = []
    for uf in uniq_freq:
        argfreq = np.where((freq == uf))[0]
        argok = argfreq[np.argsort(sc[argfreq])]
        argok = argok[::-1]
        optim += sc[argok].tolist()
        freq_optim += (np.ones((1, len(argfreq))) * uf).flatten().tolist()
    optim_ok = [0] + optim[-1 * n_scales:]
    return optim_ok, freq_optim[-1 * n_scales:]


def rf_ft_selection(trads, testds, n_scales, n_features, eval_sc, threshold=0.85, step=1):
    """
    Perform iterative feature selection using the random forest embedded feature importance as criteria.
    First, n-scales and n-features are selected based on their linear correlations and mutual information.
    Then, this set is iteratively reduced by discarding the feature having the lowest random forest feature importance.
    At each step, the model is trained again to update the feature importance ranking.

    Parameters
    ----------
    trads : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().
    testds : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().
    n_scales : int
        number of different scales to select at the begining of the process.
    n_features : int
        number of different features to select at the begining of the process.
    eval_sc : float
        scale at which to evaluate each feature's information score at the begining of the process.
    threshold : float
        accepted value of correlation between predictors.
    step : int

    Returns
    -------
    dictio_ft : dictionary
        contains the resulting predictors set and associated parameters and metrics at each iteration.
    """
    trads['features'] = feature_clean(trads['features'])
    testds['features'] = feature_clean(testds['features'])
    scales, names, ds_names = get_scales_feats(trads)
    dictio = {'Complexity': [], 'Feats': [], 'Scales': [], 'Indices': [], 'Freq': [], 'OA': [], 'Fscore': [],
              'Confidence': [], 'Recall': [], 'Precision': [], 'Class_UA': [], 'Class_PA': [], 'Class_Fscore': [],
              'Class_confidence': [], 'Class_recall': [], 'Class_precision': [], 'Labels': np.unique(trads['labels'])}
    search_set = np.array(np.where(scales == eval_sc)[0].tolist() + np.where(scales == 0)[0].tolist())
    search_ft_ds = {'features': trads['features'][:, search_set], 'labels': trads['labels'], 'names': names[search_set]}
    sel = n_best_uncorr_ft(search_ft_ds, n_features, threshold)
    scale_search_set = []
    for sn in search_ft_ds['names'][sel]:
        scale_search_set += np.where((sn == names))[0].tolist()  # indices of selected features at all scales
    search_sc_ds = {'features': trads['features'][:, scale_search_set], 'labels': trads['labels'],
                 'names': ds_names[scale_search_set]}
    sel_sc, freq_sel_sc = n_best_uncorr_sc(search_sc_ds, n_scales, threshold)
    id_es = []
    for es in sel_sc:
        id_es += np.where(scales[scale_search_set] == es)[0].tolist()
    scales_selected = np.array(scale_search_set)[id_es]  # indices of selected features at selected scales
    idx_used = []
    for ss in sel_sc:
        idx_used += scales_selected[np.where(scales[scales_selected] == float(ss))[0]].tolist()
    reduced_tra = {'features': trads['features'][:, idx_used], 'labels': trads['labels']}
    reduced_test = {'features': testds['features'][:, idx_used], 'labels': testds['labels']}
    accuracy, fscore, confid, recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imp, classifier, lab_pred = get_acc_expe(reduced_tra, reduced_test, plot=False)
    print(search_ft_ds['names'])
    print(sel)
    dictio['Feats'].append(search_ft_ds['names'][sel])
    dictio['Scales'].append(sel_sc)
    trads['features'] = feature_clean(trads['features'])
    testds['features'] = feature_clean(testds['features'])
    argimp = np.argsort(feat_imp.flatten())
    id_sort = np.array(idx_used)[argimp]  # indices of selected predictors ranked by importance
    for i in range(0, argimp.shape[0]-step, step):
        id_select = id_sort[step:]  # predictors indices
        reduced_tr = {'features': trads['features'][:, id_select], 'labels': trads['labels']}
        reduced_te = {'features': testds['features'][:, id_select], 'labels': testds['labels']}
        accuracy, fscore, confid, recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imp, classifier, lab_pred = get_acc_expe(reduced_tr, reduced_te, plot=False)
        dictio['Complexity'].append(len(id_select))
        dictio['Indices'].append(id_select)
        dictio['Feats'].append(names[id_select])
        dictio['Scales'].append(scales[id_select])
        dictio['OA'].append(accuracy)
        dictio['Fscore'].append(fscore)
        dictio['Confidence'].append(confid)
        dictio['Recall'].append(recall)
        dictio['Precision'].append(precision)
        dictio['Class_UA'].append(uas)
        dictio['Class_PA'].append(pas)
        dictio['Class_Fscore'].append(fscores)
        dictio['Class_confidence'].append(confc)
        dictio['Class_recall'].append(recalls)
        dictio['Class_precision'].append(precisions)
        argimp = np.argsort(feat_imp.flatten())
        id_sort = np.array(id_select)[argimp]
    return dictio


def get_n_optimal_sc_ft(train_ds, test_ds, n_scales, n_features, eval_sc, threshold):
    """
    Get the best n_features and n_scales for classification based on inter-feature correlation and information score.

    Parameters
    ----------
    train_ds : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().
    test_ds : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().
    n_scales : int
        number of different scales to select.
    n_features : int
        number of different features to select.
    eval_sc : float
        scale at which to evaluate each feature's information score.
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    dictio_ft : dictionary
        contains the resulting predictors set and associated parameters and metrics.

         - 'Feats': feature list
         - 'Scales': scale list
         - 'feat_imp': feature importance values
         - 'Indices': indices of the selected features in the initial array of features
         - 'Freq': number of votes obtained by each selected scale
         - 'OA': Overall Accuracy of classifier
         - 'Fscore': F1-score (averaged on all classes)
         - 'Confidence': Confidence (averaged on all classes)
         - 'Recall': Recall (averaged on all classes)
         - 'Precision': Precision (averaged on all classes)
         - 'Class_UA': User's accuracies (per class)
         - 'Class_PA': Producer's accuracies (per class)
         - 'Class_Fscore': F1-score per class
         - 'Class_confidence': class confidence
         - 'Class_recall': Recall per class
         - 'Class_precision': Precision per class
         - 'Labels': labels
    """

    scales, names, ds_names = get_scales_feats(train_ds)
    search_set = np.array(np.where(scales == eval_sc)[0].tolist() + np.where(scales == 0)[0].tolist())
    search_ft_ds = {'features': train_ds['features'][:, search_set], 'labels': train_ds['labels'], 'names': names[search_set]}
    sel = n_best_uncorr_ft(search_ft_ds, n_features, threshold)
    scale_search_set = []
    for sn in search_ft_ds['names'][sel]:
        scale_search_set += np.where((sn == names))[0].tolist()
    search_sc_ds = {'features': train_ds['features'][:, scale_search_set],
                    'labels': train_ds['labels'],
                    'names': ds_names[scale_search_set]}
    sel_sc, freq_sel_sc = n_best_uncorr_sc(search_sc_ds, n_scales, threshold)
    id_es = []
    for es in sel_sc:
        print(es)
        id_es += np.where(scales[scale_search_set] == es)[0].tolist()
    scales_selected = np.array(scale_search_set)[id_es]
    idx_used = []
    for ss in sel_sc:
        idx_used += scales_selected[np.where(scales[scales_selected] == float(ss))[0]].tolist()
    reduced_tra = {'features': train_ds['features'][:, idx_used], 'labels': train_ds['labels']}
    reduced_test = {'features': test_ds['features'][:, idx_used], 'labels': test_ds['labels']}
    accuracy, fscore, confid, recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imp, classifier, labels_pred = get_acc_expe(reduced_tra, reduced_test, plot=False)
    dictio = {'Feats': ds_names[idx_used], 'Scales': np.array(scales[idx_used]), 'feat_imp': feat_imp,
              'Indices': np.array(idx_used), 'Freq': np.array(freq_sel_sc), 'OA': accuracy, 'Fscore': fscore,
              'Confidence': confid, 'Recall': recall, 'Precision': precision,
              'Class_UA': np.array(uas), 'Class_PA': np.array(pas), 'Class_Fscore': np.array(fscores),
              'Class_confidence': np.array(confc), 'Class_recall': np.array(recalls),
              'Class_precision': np.array(precisions), 'Labels': np.array(labels)}
    return dictio


def get_best_rf_select_iter(dictio_rf_select, trads, testds, wait, threshold):
    """
    Get an optimized set of features and scales by analyzing the variations of OA or oob-score when performing
    random forest feature importance-based iterative selection.

    Parameters
    ----------
    dictio_rf_select : dictionary
        obtained when performing rf_ft_selection.
    trads : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().
    testds : dictionary
        data dictionary containing features, labels, names obtained with load_sbf_features().
    wait : int
        number of iterations to take into account for monitoring.
    threshold : float
        accepted value of OA variance within wait period.

    Returns
    -------
    dictio_results : dictionary
        contains the resulting predictors set and associated parameters and metrics.

        - 'Best_it': best iteration
        - 'Features': optimized set of features
        - 'Scales': scales related to the optimized features
        - 'Feat_names': feature names (maybe redundant with 'Features')
        - 'Feat_imp_mean': mean of feature importance
        - 'Scales_name': scale names
        - 'Scales_freq': scale frequency (per scale)
        - 'Scales_imp': scale importance (per scale)
        - 'OA': Overall Accuracy,
        - 'Fscore': F1-score
        - 'Confid': confidence
        - 'Recall': recall,
        - 'Precision': precision,
        - 'UAs': User's accuracies (per class)
        - 'PAs': Producer's accuracies (per class)
        - 'Class_fscores': F1-score per class
        - 'Class_conf': confidence per class
        - 'labels': labels

    classifier : sklearn.ensemble.RandomForestClassifier
        theoretically optimal classifier (trained only with the selected features/scales).
    """
    # data = np.load(dictio_rf_select, allow_pickle=True).flat[0]
    data = dictio_rf_select
    oa = data['OA']
    feats = data['Feats'][1:]
    scales = data['Scales'][1:]
    indexes = data['Indices']
    bi = -1
    best_oa = oa[0]
    near = False
    for i in range(wait, len(oa)-wait, 1):
        var = max(best_oa, np.max(oa[i-wait:i])) - min(best_oa, np.min(oa[i-wait:i]))
        if np.abs(var) <= 0.01:
            if near:
                if np.abs(np.max(oa[i - wait:i]) - best_oa) < threshold:
                    best_oa = np.max(oa[i - wait:i])
                    bi = i - wait + np.argmax(oa[i - wait:i])
            else:
                best_oa = np.max(oa[i - wait:i])
                bi = i - wait + np.argmax(oa[i - wait:i])
        else:
            near = True
            diffoa = oa[i - wait:i] - best_oa
            last_best = np.where(np.abs(diffoa) < threshold)[0]
            if len(last_best) > 0:
                best_oa = oa[i - wait + last_best[-1]]
                bi = i - wait + last_best[-1]
    final_feats = feats[bi]
    final_scales = scales[bi]
    final_idx = indexes[bi]
    reduced_tr = {'features': trads['features'][:, final_idx], 'labels': trads['labels']}
    reduced_te = {'features': testds['features'][:, final_idx], 'labels': testds['labels']}
    accuracy, fscore, confid, recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imp, classifier, labels_pred = get_acc_expe(reduced_tr, reduced_te)
    features = np.unique(final_feats)
    feature_imptces = []
    for f in np.unique(final_feats):
        where = np.where(final_feats == f)[0]
        mean_imp = np.mean(feat_imp[where])
        feature_imptces.append(mean_imp)
    scales_imptces = []
    for s in np.unique(final_scales):
        where = np.where(final_scales == s)[0]
        mean_imp = np.mean(feat_imp[where])
        scales_imptces.append(mean_imp)
    echelles, freq_echelles = np.unique(final_scales, return_counts=True)
    fnames = []
    for k in range(len(final_feats)):
        fname = str(final_feats[k])+str(final_scales[k])
        fnames.append(fname)
    labels = np.unique((reduced_te['labels']))
    cn = []
    for l in labels:
        cn.append(cc_3dmasc.classes[int(l)])
    for i in range(len(labels)):
        print(labels[i], cn[i], pas[i])
    dictio_results = {'Best_it': len(oa) - bi, 'Features': np.array(final_feats), 'Scales': np.array(final_scales),
                      'Feat_names': np.array(features), 'Feat_imp_mean': np.array(feature_imptces),
                      'Scales_name': np.array(echelles), 'Scales_freq': np.array(freq_echelles),
                      'Scales_imp': scales_imptces, 'OA': accuracy, 'Fscore': fscore, 'Confid': confid,
                      'Recall': recall, 'Precision': precision, 'UAs': np.array(uas), 'PAs': np.array(pas),
                      'Class_fscores': np.array(fscores), 'Class_conf': np.array(confc), 'labels': labels}
    return dictio_results, classifier
