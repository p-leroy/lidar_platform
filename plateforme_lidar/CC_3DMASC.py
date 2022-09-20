# coding: utf-8
# Paul Leroy
# Baptiste Feldmann

from . import cloudcompare, PySBF, utils
import sklearn.metrics as metrics
import numpy as np
import os


def compute_features(query0_params, workspace, params, training_file):
    """
    Call 3DMASC plugin (from CloudCompare) to compute geometric features
    
    Parameters
    ----------
    query0_params : list of string parameters [QUERY_0,EXPORT_FMT,shiftname]
    workspace : str, absolutepath to directory
    params : dict of files to use 3DMASC, [PCX,PC1,PC2,CTX]
    training_file : str, parameters file for 3DMASC
    """
    for i in params.keys():
        if not os.path.exists(workspace + params[i]):
            raise OSError("File " + params[i] + " not found !")

    query = utils.QUERY_0[query0_params[0]] + utils.EXPORT_FMT[query0_params[1]]
    for i in params.keys():
        query += " -o -global_shift "+utils.SHIFT[query0_params[2]] + " " + workspace + params[i]

    query += " -3dmasc_classify -only_features " + training_file + ' "'
    compt = 1
    for i in params.keys():
        query += i + '=' + str(compt) + ' '
        compt += 1

    query = query[0:-1] + '" -save_clouds'
    utils.run(query, sleeping=2)
    
    today = utils.DATE()
    if query0_params[1] == "SBF":
        cloudcompare.last_file(workspace + "_".join([params["PCX"][0:-4], "WITH_FEATURES", today.date, "*.sbf"]),
                               params["PCX"][0:-4]+"_features.sbf")
        cloudcompare.last_file(workspace + "_".join([params["PCX"][0:-4], "WITH_FEATURES", today.date, "*.sbf.data"]),
                               params["PCX"][0:-4]+"_features.sbf.data")
        for i in params.keys():
            tempfile = cloudcompare.last_file(workspace+"_".join([params[i][0:-4], today.date, "*.sbf"]))
            os.remove(tempfile)
            tempfile = cloudcompare.last_file(workspace+"_".join([params[i][0:-4], today.date, "*.sbf.data"]))
            os.remove(tempfile)
    
    elif query0_params[1] == "LAS":
        cloudcompare.last_file(workspace + "_".join([params[i][0:-4], "WITH_FEATURES", today.date, "*.laz"]),
                               params["PCX"][0:-4]+"_features.laz")
        for i in params.keys():                
            tempfile = cloudcompare.last_file(workspace + "_".join([params[i][0:-4], today.date, "*.laz"]))
            os.remove(tempfile)

    else:
        raise NotImplementedError("Features can only be exported in SBF or LAS format for now")


def load_features(pcx_filepath, training_filepath, labels=False):
    """
    Loading computed features after using 3DMASC plugin

    Parameters
    ---------
    pcx_filepath : str, absolute path to core-points file
    training_filepath : str, parameters file for 3DMASC
    labels : bool (default=False), in case of training model you need the labels

    Returns
    --------
    data : dict,
         'features' : numpy.array of computed features
         'names' : list of str, name of each column feature
         'labels' : list of int, class labels
    """
    f = PySBF.File(pcx_filepath)
    ptsCloud = f.points
    names = f.scalarNames
    del f

    tab = np.loadtxt(training_filepath[0:-4]+"_feature_sources.txt",str)
    list_sf = []
    for i in tab:
        name = i.split(sep=':')[1].lower()
        if name in utils.convention.keys():
            list_sf += [utils.convention[name]]
        else:
            list_sf += [name]
    idx_select = list(np.sort([names.index(i) for i in list_sf]))

    data = {"features": ptsCloud[:, idx_select], "names": list(np.array(names)[idx_select])}
    if labels:
        data["labels"] = ptsCloud[:, names.index('classification')]
    return data


def accuracy_score(labels_true,labels_pred,ind_confid=[]):
    labels = np.unique(labels_true)
    tab = metrics.confusion_matrix(labels_true, labels_pred, labels)
    liste_accuracy = []
    for i in range(0, len(labels)):
        liste_accuracy += [tab[i, i] / np.sum(tab[:, i])]
    return liste_accuracy, labels


def confidence_score(labels_pred,ind_confid):
    labels = np.unique(labels_pred)
    liste_ind = []
    for i in labels:
        temp=labels_pred == i
        liste_ind += [np.percentile(ind_confid[temp], 25)]
    return liste_ind,labels


def classif_report(labels_true, labels_pred, ind_confid=None, save=False):
    labels = np.unique(labels_true)
    tab = metrics.confusion_matrix(labels_true, labels_pred, labels)
    affiche = "Matrice de confusion :\n\t"
    for i in labels:
        affiche += str(i) + "\t"
    affiche += "\n"
    for i in range(0,len(labels)):
        affiche += str(labels[i]) + "\t"
        for c in range(0,len(labels)):
            affiche += str(tab[i,c]) + "\t"
        affiche += "\n"

    result_accuracy = accuracy_score(labels_true, labels_pred)
    if len(ind_confid) > 0:
        result_confid = confidence_score(labels_pred, ind_confid)
    else:
        result_confid = np.array([0]*len(np.unique(labels_pred)))
    
    affiche2 = "Classes     \t"
    for i in result_accuracy[1]:
        affiche2 += str(i)+"\t"
    affiche2 += "\nPrecision  \t"
    for i in result_accuracy[0]:
        affiche2 += str(np.round_(i * 100, 1)) + "\t"
    affiche2 += "\nInd_conf 25%\t"
    for i in result_confid[0]:
        affiche2 += str(np.round_(i * 100, 1)) + "\t"
    affiche2 += "\n"
    
    test = (labels_true == labels_pred)
    val_true = len([i for i, X in enumerate(test) if X])
    print(affiche)
    print(affiche2)
    print("Pourcentage valeur Vrai : %.1f%%" % (val_true / len(test) * 100))
    print("indice Kappa : " + str(metrics.cohen_kappa_score(labels_true, labels_pred)))
    print(metrics.classification_report(labels_true, labels_pred, labels))
    
    if save :
        f = open(save, "w")
        print(affiche + "\n", file=f)
        print(affiche2, file=f)
        print("Pourcentage valeur Vrai : %.1f%%" %(val_true / len(test) * 100), file=f)
        print("indice Kappa : " + str(metrics.cohen_kappa_score(labels_true,labels_pred)), file=f)
        print(metrics.classification_report(labels_true, labels_pred, labels), file=f)
        f.close()

        
def feature_importance_analysis(tab, nb_scales):
    """
    Feature Importance Analysis
    (by features and by scales)
    """
    if len(tab) % nb_scales == 0:
        nb_feat = int(len(tab) / nb_scales)
    else:
        raise ValueError("Length of your array doesn't match exactly number of scales")

    by_feat = []
    for i in range(0, nb_feat):
        p = np.arange(0, nb_scales) + (i * nb_scales)
        by_feat += [np.sum(tab[p])]

    by_scales = []
    for i in range(0, nb_scales):
        p = [nb_scales * m + i for m in range(0, nb_feat)]
        by_scales += [np.sum(tab[p])]
        
    print("Feature Importances Analysis :\n\t-By features :")
    print(by_feat)
    print("\t-By scales :")
    print(by_scales)


def correlation(data, names, save=""):
    func = getattr(getattr(__import__("scipy"), "stats"), "pearsonr")
    list_ = []
    for i in range(0, len(data[0, :])):
        temp = []
        for c in range(0, len(data[0, :])):
            if c < i + 1:
                temp += [-2]
            else:
                result = func(data[:, i], data[:, c])[0]
                temp += [round(result, 4)]
                if result > 0.75:
                    print("Correlation between : %s and %s (%.4f)" % (names[i], names[c], result))
        list_ += [temp]
    if len(save) > 0:
        np.savetxt(save, list_, fmt='%.4f', delimiter=";")
