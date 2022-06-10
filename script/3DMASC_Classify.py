import plateforme_lidar as pl
import numpy as np
import pickle,glob,os
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel,delayed
import time
import importlib
importlib.reload(pl)

#----Classification----#
def computeFeatures(workspace,PCX_filename,params_CC,params_features):
    params_training={"PC1":"_".join(["PC1"]+PCX_filename.split("_")[1::]),
                     "PCX":PCX_filename,
                     "CTX":"_".join(["CTX"]+PCX_filename.split("_")[1::])}
    
    pl.CC_3DMASC.compute_features(params_CC,workspace,params_training,params_features)
    os.rename(workspace+PCX_filename[0:-4]+"_features.sbf",workspace+"features/"+PCX_filename[0:-4]+"_features.sbf")
    os.rename(workspace+PCX_filename[0:-4]+"_features.sbf.data",workspace+"features/"+PCX_filename[0:-4]+"_features.sbf.data")

def classify(workspace,filename,model,features_file):
    dictio=pl.CC_3DMASC.load_features(workspace+"features/"+filename[0:-4]+"_features.sbf",features_file)
    #Normalize by (0,1) and replace nan by -1
    data=MinMaxScaler((0,1)).fit_transform(dictio['features'])
    data=np.nan_to_num(data,nan=-1)
    
    labels_pred=model.predict(data)
    confid_pred=model.predict_proba(data)
    confid_pred=np.max(confid_pred,axis=1)
    lasdata=pl.lastools.ReadLAS(workspace + filename)
    
    lasdata.classification=labels_pred
    #print(np.shape(lasdata))
    #print(np.shape(data))
    extra=[(("ind_confid", "float32"), np.round(confid_pred * 100,decimals=1))]
    pl.lastools.WriteLAS(workspace + filename[0:-4] + "_class.laz", lasdata, format_id=1, extraField=extra)

workspace=r'G:\RENNES1\Loire_totale_automne2019\Loire_Briare-Sully-sur-Loire\05-Traitements\C3\classification\bathy\haute_resolution'+'//'
liste=glob.glob(workspace+"PCX_*00.laz")
liste_noms=[os.path.split(i)[1] for i in liste]
print("%i files found !" %(len(liste_noms)))

classifier={
    "path":r'G:\RENNES1\BaptisteFeldmann\Python\training\Loire\juin2019\classif_C3_withSS'+'//',
    "features_file":"Loire_20190529_C3_params_v3.txt",
    "model":"Loire_Rtemus2019_C3_HR_model_v3.pkl"}
queryCC_param=['standard','SBF','Loire45-4']

#computeFeatures(workspace,"PCX_660000_6739000.laz",queryCC_param,classifier["path"]+classifier["features_file"])


deb=time.time()
for i in liste_noms:
    print(i+" "+str(liste_noms.index(i)+1)+"/"+str(len(liste_noms)))
    if not os.path.exists(workspace+"features/"+i[0:-4]+"_features.sbf"):
        computeFeatures(workspace,i,queryCC_param,classifier["path"]+classifier["features_file"])
    print("================================")
print("Time duration: %.1f sec" %(time.time()-deb))

infile=open(classifier["path"]+classifier["model"],"rb")
tree=pickle.load(infile)
infile.close()
tree.verbose=1
tree.n_jobs=50

#classify(workspace,"PCX_660000_6739000.laz",tree,classifier["path"]+classifier["features_file"])

for i in liste_noms:
    print(i)
    if not os.path.exists(workspace+i[0:-4]+"_class.laz"):
        classify(workspace,i,tree,classifier["path"]+classifier["features_file"])

#Parallel(n_jobs=10,verbose=2)(delayed(classify)(workspace,i,tree,classifier["path"]+classifier["features_file"]) for i in liste_noms)
#=============================#




