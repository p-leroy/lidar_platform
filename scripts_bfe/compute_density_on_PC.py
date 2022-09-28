import plateforme_lidar as pl
from joblib import Parallel,delayed
import glob
import numpy as np

def defineGrid(step,sizeX,sizeY,x0,y0):
    #lower left
    tab=[]
    for i in range(x0,x0+sizeX,step):
        for c in range(y0,y0+sizeY,step):
            tab+=[[i+0.5*step,c+0.5*step]]
    return np.array(tab)

def func(filepath):
    data= tools.lastools.readLAS_laspy(filepath)
    lowerleft=np.int_(np.amin(data.XYZ[:,0:2],axis=0))
    sizes=np.int_(np.amax(data.XYZ[:,0:2],axis=0))-lowerleft
    grid=defineGrid(1,sizes[0],sizes[1],*lowerleft)
    result=pl.calculs.compute_density(data.XYZ[:, 0:2], grid, 0.5, np.inf)
    np.savez_compressed(filepath[0:-4]+"_density.npz",result)

workspace=r'G:\RENNES1\Loire_octobre2020_Rtemus\05-Traitements\C2\classification\final'+'//'
listFilenames=glob.glob(workspace+"*.laz")

Parallel(n_jobs=30,verbose=2)(delayed(func)(i) for i in listFilenames)

all_tab=[]
for i in listFilenames:
    with np.load(i[0:-4]+"_density.npz") as f:
        tab=f[f.files[0]]
    select=tab>0
    all_tab+=[tab[select]]
    
all_tab=np.concatenate(all_tab)
print(np.percentile(all_tab,10))
#[os.remove(i) for i in glob.glob(workspace+"*_density.npz")]
