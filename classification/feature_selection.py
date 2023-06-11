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
    Get the list of scales and features present in the file.

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names.

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
    Get the percentage of NaN values for each feature.

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names.

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

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names.

    Returns
    -------
    dictio_ft : dictionary
        dictionary containing the name of each feature and the associated score value.
    """
    ds_cleaned = feature_clean(ds['features'])
    mi = sklearn.feature_selection.mutual_info_classif(ds_cleaned, ds['labels'])
    dictio_ft = {'Features': ds['names'], 'MutualInfo': mi}
    return dictio_ft


def correlation_score_filter(features_set, features_score, threshold):
    """
    Prune a set of predictors by keeping only the most informative elements among correlated pairs.

    Parameters
    ----------
    features_set : numpy array (n_points x n_predictors)
        array containing the value of each predictor evaluated for each point.
    features_score : numpy array (n_predictors x 1)
        array containing the information score of each predictor.
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    select : list(int)
        list of the indices of the selected features in the original features_set array.
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


def correlation_selection_filter(features, features_set, ft_select, threshold):
    """
    Check compatibility of features considering their linear correlation.

    Parameters
    ----------
    features : numpy array (n_points x n_predictors)
        array containing the value of each predictor for each point.
    features_set : list(int)
        indices of the elements to consider for selection (candidates to evaluate).
    ft_select : numpy array (n_points x n_selected)
        array containing the value of each predictor that already passed the selection.
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    valides : list(int)
        list of the indices of the selected features in the original ds['features'] array.
    """
    feats_df = pd.DataFrame(features[:, features_set])
    corr_mat = feats_df.corr().to_numpy()
    half = np.abs(corr_mat)
    for i in range(corr_mat.shape[0]):
        half[i, ] = 0
    select = np.arange(0, len(ft_select), 1)  # indices of already selected features
    cor = np.where((half >= threshold))  # indices of correlated features
    todel = []
    for s in select:
        todel += cor[1][np.where((cor[0] == s))[0].tolist()].tolist()  # features correlated to validated features set
        todel += cor[0][np.where((cor[1] == s))[0].tolist()].tolist()  # features correlated to validated features set
    invalides = select.tolist() + np.unique(np.array(todel)).tolist()  # indices of invalid features
    valides = np.delete(features_set, invalides)
    return valides.tolist()


def selection(ft_all, ft_select, ft_score, nf, threshold):
    """
    Among features that passed the correlation and information score criteria, select those that are compatible
    with previously elected features.

    Parameters
    ----------
    ft_all : numpy array (n_points x n_predictors)
        array containing the value of each predictor for each point.
    ft_select : numpy array (n_points x n_selected)
        array containing the value of each predictor that already passed the selection.
    ft_score : numpy array (n_predictors x 1)
        array containing the information score of each predictor.
    nf : int
        number of different features to select.
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    ft_select : list(int)
        list of the indices of the selected features in the original ds['features'] array.
    """
    n_select = len(ft_select)
    argsort_sc = np.argsort(ft_score)
    possib = argsort_sc[-1 * nf:]
    nb_tested = nf
    while n_select < nf and possib.shape[0] > 0:
        possib = argsort_sc[:-1 * nb_tested]
        rem = nf - n_select
        test = ft_select + possib[-1 * rem:].tolist()  # indices of selected features and best features
        compatibles = correlation_selection_filter(ft_all, test, ft_select, threshold)  # all selected features
        valides_id = correlation_score_filter(ft_all[:, compatibles], ft_score[compatibles], threshold)
        valides = np.array(compatibles)[valides_id]
        ft_select += valides.tolist()
        n_select = len(ft_select)
        nb_tested += rem
    return ft_select


def select_best_feature_set(ds, nf, corr_threshold):
    """
    Select best feature set for a given dataset, using information score and correlation between predictors.

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names.
    nf : int
        number of different features to select.
    corr_threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    select : list(int)
        list of the indices of the selected features in the original ds['features'] array.
    """
    ft_score = info_score(ds)['MutualInfo']
    possib = np.argsort(ft_score)[-1 * nf:]
    select_id = correlation_score_filter(ds['features'][:, possib], ft_score[possib], corr_threshold).tolist()
    select = np.array(possib)[select_id]
    if len(select.tolist()) != nf:
        select = selection(ds['features'], select.tolist(), ft_score, nf, corr_threshold)
    return select


def select_best_scales(ds, ns, corr_threshold):
    """
    Select best feature computation scales for a given dataset, using information score, correlation between predictors,
    and a voting process.

    Parameters
    ----------
    ds : dictionary
        data dictionary containing features, labels, names.
    ns : int
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
        result = select_best_feature_set(search_ds, ns, corr_threshold)
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
    optim_ok = [0] + optim[-1 * ns:]
    return optim_ok, freq_optim[-1 * ns:]


def embedded_f_selection(trads, testds, nscales, nfeats, eval_sc, threshold=0.85, step=1):
    """
    Perform iterative feature selection using the RF embedded feature importance as criteria.

    Parameters
    ----------
    trads : dictionary
        data dictionary containing features, labels, names.
    testds : dictionary
        data dictionary containing features, labels, names.
    nscales : int
        number of different scales to select at the begining of the process.
    nfeats : int
        number of different features to select at the begining of the process.
    eval_sc : float
        scale at which to evaluate each feature's information score at the begining of the process.
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    dictio_ft : dictionary
        dictionary containing the resulting predictors set and associated parameters and metrics at each iteration.
    """
    trads['features'] = feature_clean(trads['features'])
    testds['features'] = feature_clean(testds['features'])
    scales, names, ds_names = get_scales_feats(trads)
    dictio = {'Complexity': [], 'Feats': [], 'Scales': [], 'Indices': [], 'Freq': [], 'OA': [], 'Fscore': [],
              'Confidence': [], 'Recall': [], 'Precision': [], 'Class_UA': [], 'Class_PA': [], 'Class_Fscore': [],
              'Class_confidence': [], 'Class_recall': [], 'Class_precision': [], 'Labels': np.unique(trads['labels'])}
    search_set = np.array(np.where(scales == eval_sc)[0].tolist() + np.where(scales == 0)[0].tolist())
    search_ft_ds = {'features': trads['features'][:, search_set], 'labels': trads['labels'], 'names': names[search_set]}
    sel = select_best_feature_set(search_ft_ds, nfeats, threshold)
    scale_search_set = []
    for sn in search_ft_ds['names'][sel]:
        scale_search_set += np.where((sn == names))[0].tolist()  # indices of selected features at all scales
    search_sc_ds = {'features': trads['features'][:, scale_search_set], 'labels': trads['labels'],
                 'names': ds_names[scale_search_set]}
    sel_sc, freq_sel_sc = select_best_scales(search_sc_ds, nscales, threshold)
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


def get_selection(trads, testds, nscales, nfeats, eval_sc, threshold):
    """
    Get the best nfeats and nscales for classification based on inter-feature correlation and information score.

    Parameters
    ----------
    trads : dictionary
        data dictionary containing features, labels, names.
    testds : dictionary
        data dictionary containing features, labels, names.
    nscales : int
        number of different scales to select.
    nfeats : int
        number of different features to select.
    eval_sc : float
        scale at which to evaluate each feature's information score.
    threshold : float
        accepted value of correlation between predictors.

    Returns
    -------
    dictio_ft : dictionary
        dictionary containing the resulting predictors set and associated parameters and metrics.
    """
    scales, names, ds_names = get_scales_feats(trads)
    search_set = np.array(np.where(scales == eval_sc)[0].tolist() + np.where(scales == 0)[0].tolist())
    search_ft_ds = {'features': trads['features'][:, search_set], 'labels': trads['labels'], 'names': names[search_set]}
    sel = select_best_feature_set(search_ft_ds, nfeats, threshold)
    scale_search_set = []
    for sn in search_ft_ds['names'][sel]:
        scale_search_set += np.where((sn == names))[0].tolist()
    search_sc_ds = {'features': trads['features'][:, scale_search_set], 'labels': trads['labels'],
                    'names': ds_names[scale_search_set]}
    sel_sc, freq_sel_sc = select_best_scales(search_sc_ds, nscales, threshold)
    id_es = []
    for es in sel_sc:
        print(es)
        id_es += np.where(scales[scale_search_set] == es)[0].tolist()
    scales_selected = np.array(scale_search_set)[id_es]
    idx_used = []
    for ss in sel_sc:
        idx_used += scales_selected[np.where(scales[scales_selected] == float(ss))[0]].tolist()
    reduced_tra = {'features': trads['features'][:, idx_used], 'labels': trads['labels']}
    reduced_test = {'features': testds['features'][:, idx_used], 'labels': testds['labels']}
    accuracy, fscore, confid, recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imp, classifier, labels_pred = get_acc_expe(reduced_tra, reduced_test, plot=False)
    dictio = {'Feats': ds_names[idx_used], 'Scales': np.array(scales[idx_used]), 'feat_imp': feat_imp,
              'Indices': np.array(idx_used), 'Freq': np.array(freq_sel_sc), 'OA': accuracy, 'Fscore': fscore,
              'Confidence': confid, 'Recall': recall, 'Precision': precision,
              'Class_UA': np.array(uas), 'Class_PA': np.array(pas), 'Class_Fscore': np.array(fscores),
              'Class_confidence': np.array(confc), 'Class_recall': np.array(recalls),
              'Class_precision': np.array(precisions), 'Labels': np.array(labels)}
    return dictio


def get_best_iter(dictio_rf_select, trads, testds, wait, threshold):
    """
    Get the theoretically optimal predictor set for classification by monitoring OA drops when performing embedded
    feature selection.

    Parameters
    ----------
    dictio_rf_select : dictionary
        dictionary obtained when performing embedded_rf_selection.
    trads : dictionary
        data dictionary containing features, labels, names.
    testds : dictionary
        data dictionary containing features, labels, names.
    wait : int
        number of iterations to take into account for monitoring.
    threshold : float
        accepted value of OA variance within wait period.

    Returns
    -------
    dictio_ft : dictionary
        dictionary containing the resulting predictors set and associated parameters and metrics.
    classifier :  sklearn RandomForestClassifier
        theoretically optimal classifier.
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
    return dictio_results, classifier, reduced_te
