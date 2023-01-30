# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:33:50 2022

@author: 33628
"""
import numpy as np
import pandas as pd
from default_pc_classif import *
from cc_3dmasc import get_acc_expe, feature_clean
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import plotly.graph_objects as go
from confid_errors import *


def get_scales_feats(ds):
    scales = []
    names = []
    for i in ds['names']:
        name = i
        if 'knn' not in i:
            split = i.split(sep='@')
            if len(split) > 1:
                scale = float(split[1])
                scales += [scale]
            else:
                scales += [0]
            names += [name.split('@')[0]]
        else:
            scales += [0]
            names += [name]
    return np.array(scale), np.array(name), np.array(ds['names'])


def multivariate_analysis(features, target, out, plot=False, save=True):
    data = np.hstack((features, target.reshape((-1, 1))))
    data = pd.DataFrame(data)
    fig = plt.figure(constrained_layout=True)
    sns.set(rc={'figure.figsize': (23.4, 16.54)})
    ax = sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='magma_r')
    ax.set_title(out)
    if plot:
        plt.show()
    if save:
        plt.savefig(out+'.jpeg')


def nan_percentage(features, names, plot=True, save=False, out=''):
    scales, names, ds_names = get_scales_feats(trads)
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
                feat_data = features[:,sorted_indices[f]:]
                scales_feat = scales[sorted_indices[f]:]
            if len(scales_feat) > 2:
                ok = True
            if ok:
                for j in range(feat_data.shape[1]):
                    percent.append(len(np.where(np.isnan(feat_data[:, j]))[0])/feat_data.shape[0])

                dictio_ft = {'Scales': scales_feat, 'Percentage': percent}
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(name='Percentage',  x=dictio_ft['Scales'], y=dictio_ft['Percentage'], mode='lines', line = dict(color='darkolivegreen', width=4)),
                )
                fig.update_layout(
                    title_text="<b>Percentage of NaN values depending on the scale of the neighborhood<b>"
                )
                fig.update_xaxes(tickangle=45, showgrid=True, type='category')
                fig.update_yaxes(title_text="<b>Value</b>")
                if plot:
                    fig.write_html('NaNpercentage_' + out +'.html', auto_open=True)
                if save:
                    fig.write_image('NaNpercentage_' + out +'.jpg', scale=3)


def univariate_analysis(features, target, names):
    features_cleaned = feature_clean(features)
    mi = sklearn.feature_selection.mutual_info_classif(features_cleaned, target)
    dictio_ft = {'Features': np.array(names), 'MutualInfo': mi}
    return dictio_ft


def correlation_score_filter(features_set, features_score, threshold, save=True, plot=False):
    nb_feats = features_set.shape[1]
    feats_df = pd.DataFrame(features_set)
    corr_mat = feats_df.corr().to_numpy()
    fig = plt.figure(constrained_layout=True)
    sns.set(rc={'figure.figsize': (23.4, 16.54)})
    ax = sns.heatmap(corr_mat, annot=True, fmt='.4f', cmap='magma_r')
    plt.title('Correlation map')
    if plot:
        plt.show()
    half = np.abs(corr_mat)
    for i in range(corr_mat.shape[0]):
        half[i, i] = 0
    select = np.arange(0, nb_feats, 1)
    cor = np.where((half >= threshold))
    todel = []
    if len(cor[0]) > 0:
        for f in range(len(cor[0])):
            comp = [cor[0][f], cor[1][f]]
            todel.append(comp[np.argmin(np.array(features_score)[comp])])
    select = np.delete(select, todel, axis=0)
    return select #indice des features selectionnees


def correlation_selection_filter(features, features_set, ft_select, threshold):
    feats_df = pd.DataFrame(features[:, features_set])
    corr_mat = feats_df.corr().to_numpy()
    half = np.abs(corr_mat)
    for i in range(corr_mat.shape[0]):
        half[i, ] = 0
    select = np.arange(0, len(ft_select), 1) #indice dans la matrice des feats deja validees
    cor = np.where((half >= threshold)) #indice de features correlees entre elles
    todel = []
    for s in select:
        todel += cor[1][np.where((cor[0] == s))[0].tolist()].tolist() #features correlees a la feature deja validee
        todel += cor[0][np.where((cor[1] == s))[0].tolist()].tolist() #features correlees a la feature deja validee
    invalides = select.tolist() + np.unique(np.array(todel)).tolist() #indices dans la matrice des features invalidees
    valides = np.delete(features_set, invalides)
    return valides.tolist()


def selection(ft_all, ft_select, ft_score, names_all, nf, threshold):
    ft_select = ft_select
    n_select = len(ft_select)
    argsort_sc = np.argsort(ft_score)
    possib = argsort_sc[-1 * nf:] #si on arrive à cette fonction c'est qu'on a déjà fait une preselection avant
    nb_tested = nf
    while n_select < nf and possib.shape[0] > 0:
        possib = argsort_sc[:-1 * nb_tested]
        rem = nf - n_select
        test = ft_select + possib[-1 * rem:].tolist() #indices des x meilleures features et des features déjà sélectionnees
        compatibles = correlation_selection_filter(ft_all, test, ft_select, threshold) #contient les deja select ET les nouvelles select
        valides_id = correlation_score_filter(ft_all[:, compatibles], ft_score[compatibles], names_all, threshold)
        valides = np.array(compatibles)[valides_id]
        ft_select += valides.tolist()
        n_select = len(ft_select)
        nb_tested += rem
    return ft_select


def select_best_feature_set(ft_all, labels_all, names_all, nf, corr_threshold):
    ft_score = univariate_analysis(ft_all, labels_all, names_all)['MutualInfo']
    argsort = np.argsort(ft_score)
    possib = argsort[-1 * nf:]
    select_id = correlation_score_filter(ft_all[:, possib], ft_score[possib], names_all[possib], corr_threshold).tolist()
    select = np.array(possib)[select_id]
    if len(select.tolist()) != nf:
        select = selection(ft_all, select.tolist(), ft_score, names_all, nf, corr_threshold)
    return select


def select_best_scales(ft_all_scales, labels, names_all_scales, ns, corr_threshold):
    scales, names, ds_names = get_scales_feats(trads)
    best_scales = []
    for f in np.unique(names):
        f_id = np.where((names == f))[0]
        scales_feat = ft_all_scales[:, f_id]
        labels_feat = labels
        names_feat = np.array(names_all_scales)[f_id]
        result=select_best_feature_set(scales_feat, labels_feat, names_feat, ns, corr_threshold)
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
    optim_ok = [0]
    optim_ok += optim[-1 * ns:]
    return optim_ok, freq_optim[-1 * ns:]


def embedded_f_selection(trads, testds, features_file, nscales, nfeats, eval_sc, threshold):
    trads['features'], trads['labels'] = feature_clean(trads['features'], trads['labels'])
    testds['features'], testds['labels'] = feature_clean(testds['features'], testds['labels'])
    scales, names, ds_names = get_scales_feats(trads)
    Nsr = []
    OA = []
    FS = []
    conf = []
    Rapp = []
    Prec = []
    UA_c = []
    PA_c = []
    FS_c = []
    Conf_c = []
    Rapp_c = []
    Prec_c = []
    Lab = []
    Feats = []
    Scales=[]
    Scales_freq = []
    Indices = []
    print('selecting features...')
    search_set = np.where(scales == eval_sc)[0].tolist()
    search_set += np.where(scales == 0)[0].tolist()
    search_set = np.array(search_set)
    search_names = names[search_set]
    sel = select_best_feature_set(trads['features'][:, search_set], trads['labels'], ds_names[search_set], nfeats, threshold)
    print('selecting scales...')
    scale_search_set = []
    for sn in search_names[sel]:
        scale_search_set += np.where((sn == names))[0].tolist() #vrais indices de features selectionnees a toutes echelles

    sel_sc, freq_sel_sc = select_best_scales(trads['features'][:, scale_search_set], trads['labels'], ds_names[scale_search_set], nscales, threshold)
    id_es = []
    for es in sel_sc:
        id_es += np.where(scales[scale_search_set] == es)[0].tolist()
    scales_selected = np.array((scale_search_set))[id_es] #vrais indices des features selectionees aux echelles selectionnees
    idx_used = []
    for ss in sel_sc:
        idx_used += scales_selected[np.where(scales[scales_selected] == float(ss))[0]].tolist() #vrais indices des features selectionees aux echelles selectionnees
    reduced_tra = {'features': trads['features'][:, idx_used], 'labels': trads['labels']}
    reduced_test = {'features': testds['features'][:, idx_used], 'labels': testds['labels']}
    accuracy, fscore, confid, recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imp, classifier, lab_pred = get_acc_expe(reduced_tra, reduced_test)
    Feats.append(search_names[sel])
    Scales.append(sel_sc)
    trads['features'], trads['labels'] = feature_clean(trads['features'], trads['labels'])
    testds['features'], testds['labels'] = feature_clean(testds['features'], testds['labels'])

    argimp = np.argsort(feat_imp.flatten())
    id_sort = np.array(idx_used)[argimp] #vrais indices des features selectionees aux echelles selectionnees tries dans ordre croissant d importance de feature
    for i in range(argimp.shape[0]-1):
        id_select = id_sort[1:] #on enleve progressivement les moins bonnes features du point de vue de l importance
        #id_select = vrais indices de features
        reduced_tr={'features': trads['features'][:, id_select], 'labels': trads['labels']}
        reduced_te={'features': testds['features'][:, id_select], 'labels': testds['labels']}
        accuracy, fscore, confid, recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imp, classifier, lab_pred = get_acc_expe(reduced_tr, reduced_te)
        Nsr.append(len(id_select))
        Indices.append(id_select)
        Feats.append(names[id_select])
        Scales.append(scales[id_select])
        print(np.unique(scales[id_select]))
        OA.append(accuracy)
        FS.append(fscore)
        conf.append(confid)
        Rapp.append(recall)
        Prec.append(precision)
        UA_c.append(uas)
        PA_c.append(pas)
        FS_c.append(fscores)
        Conf_c.append(confc)
        Rapp_c.append(recalls)
        Prec_c.append(precisions)
        Lab.append(labels)
        Scales_freq.append(freq_sel_sc)

        argimp = np.argsort(feat_imp.flatten())
        id_sort = np.array(id_select)[argimp]

    dictio = {'Complexity': np.array(Nsr), 'Feats': np.array(Feats), 'Scales': np.array(Scales),
              'Indices': np.array(Indices), 'Freq': np.array(Scales_freq), 'OA': np.array(OA), 'Fscore': np.array(FS),
              'Confidence': np.array(conf), 'Recall': np.array(Rapp), 'Precision': np.array(Prec),
              'Class_UA': np.array(UA_c), 'Class_PA': np.array(PA_c), 'Class_Fscore': np.array(FS_c),
              'Class_confidence': np.array(Conf_c), 'Class_recall': np.array(Rapp_c),
              'Class_precision': np.array(Prec_c), 'Labels': np.array(labels)}
    return dictio


def get_selection(trads, testds, nscales, nfeats, eval_sc, threshold):
    print(threshold, eval_sc)
    scales, names, ds_names = get_scales_feats(trads)
    Ns=[]
    Nf=[]
    Nsr = []
    Nfr=[]
    Ft_imp=[]
    OA = []
    FS = []
    conf = []
    Rapp = []
    Prec = []
    UA_c = []
    PA_c = []
    FS_c = []
    Conf_c = []
    Rapp_c = []
    Prec_c = []
    Lab = []
    Feats = []
    Scales=[]
    Scales_freq=[]
    print('selecting features...')
    search_set = np.where(scales == eval_sc)[0].tolist()
    search_set += np.where(scales == 0)[0].tolist()
    search_set = np.array(search_set)
    search_names = names[search_set]
    sel = select_best_feature_set(trads['features'][:, search_set], trads['labels'], ds_names[search_set], nfeats, threshold)
    print(len(sel), ' features selected')
    print('selecting scales...')
    scale_search_set = []
    for sn in search_names[sel]:
        scale_search_set += np.where((sn == names))[0].tolist()

    sel_sc, freq_sel_sc = select_best_scales(trads['features'][:, scale_search_set], trads['labels'], ds_names[scale_search_set], nscales, threshold)
    id_es = []
    for es in sel_sc:
        print(es)
        id_es += np.where(scales[scale_search_set] == es)[0].tolist()
    scales_selected = np.array(scale_search_set)[id_es]
    idx_used = []
    print(len(sel_sc), ' scales selected')
    for ss in sel_sc:
        idx_used += scales_selected[np.where(scales[scales_selected] == float(ss))[0]].tolist()
    reduced_tra = {'features': trads['features'][:, idx_used], 'labels': trads['labels']}
    reduced_test = {'features': testds['features'][:, idx_used], 'labels': testds['labels']}
    np.save('reduced_training.npy', reduced_tra)
    np.save('red_tra_names.npy', np.array((trads['names']))[idx_used])
    np.save('reduced_test.npy', reduced_test)
    np.save('red_test_names.npy', np.array((testds['names']))[idx_used])
    accuracy, fscore, confid, recall, precision, uas, pas, fscores, confc, recalls, precisions, labels, feat_imp, classifier, labels_pred = get_acc_expe(reduced_tra, reduced_test)
    Ns.append(nscales)
    Nf.append(nfeats)
    Nsr.append(len(id_es))
    Nfr.append(len(sel))
    Ft_imp.append(feat_imp)
    Feats.append(ds_names[idx_used])
    Scales.append(scales[idx_used])
    OA.append(accuracy)
    FS.append(fscore)
    conf.append(confid)
    Rapp.append(recall)
    Prec.append(precision)
    UA_c.append(uas)
    PA_c.append(pas)
    FS_c.append(fscores)
    Conf_c.append(confc)
    Rapp_c.append(recalls)
    Prec_c.append(precisions)
    Lab.append(labels)
    Scales_freq.append(freq_sel_sc)
    print(len(idx_used))

    dictio = {'Complexity': np.array(Nsr), 'Feats': np.array(Feats), 'Scales': np.array(Scales),
              'Indices': np.array(Indices), 'Freq': np.array(Scales_freq), 'OA': np.array(OA), 'Fscore': np.array(FS),
              'Confidence': np.array(conf), 'Recall': np.array(Rapp), 'Precision': np.array(Prec),
              'Class_UA': np.array(UA_c), 'Class_PA': np.array(PA_c), 'Class_Fscore': np.array(FS_c),
              'Class_confidence': np.array(Conf_c), 'Class_recall': np.array(Rapp_c),
              'Class_precision': np.array(Prec_c), 'Labels': np.array(labels)}
    return dictio


def get_best_iter(dictio_rf_select, trads, testds, wait, threshold):
    data = np.load(dictio_rf_select, allow_pickle=True).flat[0]
    OA = data['OA']
    feats = data['Feats'][1:]
    scales = data['Scales'][1:]
    indexes = data['Indices']
    bi = -1
    best_oa = OA[0]
    near = False
    for i in range(wait, OA.shape[0]-wait, 1):
        var = max(best_oa, np.max(OA[i-wait:i])) - min(best_oa, np.min(OA[i-wait:i]))
        if np.abs(var) <= 0.01:
            if near:
                if np.abs(np.max(OA[i - wait:i]) - best_oa) < threshold:
                    best_oa = np.max(OA[i - wait:i])
                    bi = i - wait + np.argmax(OA[i - wait:i])
            else:
                best_oa = np.max(OA[i - wait:i])
                bi = i - wait + np.argmax(OA[i - wait:i])
        else:
            near = True
            diffoa = OA[i - wait:i] - best_oa
            last_best = np.where(np.abs(diffoa) < threshold)[0]
            if len(last_best) > 0:
                best_oa = OA[i - wait + last_best[-1]]
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
    fnames=[]
    for k in range(len(final_feats)):
        fname = str(final_feats[k])+str(final_scales[k])
        fnames.append(fname)
    labels = np.unique((reduced_te['labels']))
    cn = []
    for l in labels:
        cn.append(classes[int(l)])
    for i in range(len(labels)):
        print(labels[i], cn[i], pas[i])
    dictio_results = {'Best_it': OA.shape[0] - bi, 'Features': np.array(final_feats), 'Scales': np.array(final_scales),
                      'Feat_names': np.array(features), 'Feat_imp_mean': np.array(feature_imptces),
                      'Scales_name': np.array(echelles), 'Scales_freq': np.array(freq_echelles),
                      'Scales_imp': scales_imptces, 'OA': accuracy, 'Fscore': fscore, 'Confid': confid,
                      'Recall': recall, 'Precision': precision, 'UAs': np.array(uas), 'PAs': np.array(pas),
                      'Class_fscores': np.array(fscores), 'Class_conf': np.array(confc), 'labels': labels}
    return dictio_results
