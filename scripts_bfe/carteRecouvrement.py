import plateforme_lidar as PL
import numpy as np
import glob
import os
import pickle
from joblib import Parallel,delayed

import importlib
importlib.reload(PL)

class Overlap(object):
    def __init__(self,workspace,m3c2File,outName,settings=[]):
        # settings=[['standard,'LAS','Loire49-2'],rootNames,distUncertainty=0.02,m3c2Dist,lineNbLength=2]
        self.workspace=workspace
        self.m3c2File=m3c2File
        self.outName=outName
        self.CCoptions=settings[0]
        self.rootNamesDict=settings[1]
        self.filterDistUncertainty=settings[2]
        self.m3c2Dist=settings[3]
        self.lineNbLength=settings[4]
        self.keyList=list(self.rootNamesDict.keys())
        self.rootLength=len(self.rootNamesDict[self.keyList[0]][0])
        self._preprocessingStatus=False
    
    def _selectPairs(self):
        if os.path.exists(self.workspace+"comparison.pkl"):
            self.pairsDict=pickle.load(open(self.workspace+"comparison.pkl",'rb'))
        else:
            self.pairsDict=PL.calculs.select_pairs_overlap(self.workspace+"*_thin.laz",[self.rootLength,self.lineNbLength])
            pickle.dump(self.pairsDict,open(self.workspace+"comparison.pkl",'wb'))
        
    def _selectRootname(self,lineNumber):
        test=True
        compt=0
        while test:
            if int(lineNumber)<=self.keyList[compt]:
                test=False
            else:
                compt+=1
        rootname=self.rootNamesDict[self.keyList[compt]]
        return rootname[0]+lineNumber+rootname[1]

    def Preprocessing(self):
        print("[Overlap] Pre-processing...")
        self.listFiles=[]
        self.listCompareLines=[]
        self._selectPairs()
        if len(self.pairsDict.keys())==0:
            raise ValueError("Comparison dictionnary is empty")
        
        for i in self.pairsDict.keys():
            fileA=self._selectRootname(i)
            fileCorepts=fileA[0:-4]+"_thin.laz"
            for c in self.pairsDict[i]:
                fileB=self._selectRootname(c)
                fileResult=fileCorepts[0:-4]+"_m3c2_"+i+"and"+c+".laz"
                self.listFiles+=[[fileA,fileB,fileCorepts,fileResult]]
                self.listCompareLines+=[i+"_"+c]        
        self._preprocessingStatus=True
        print("[Overlap] Pre-processing done !")

    def Processing(self):
        if self._preprocessingStatus:
            for i in range(0,len(self.listFiles)):
                print("#=========================#")
                print("Comparison : "+self.listCompareLines[i])
                print("#=========================#")
                self.Comparison(*self.listFiles[i])
                    
            print("[Overlap] M3C2 analyzing...")
            Parallel(n_jobs=20,verbose=1)(delayed(self.WritingFile)(self.workspace+self.listFiles[i][-1]) for i in range(0,len(self.listFiles)))
            query="lasmerge -i "+self.workspace+"*_clean.laz -o "+self.workspace+self.outName
            utils.run(query)
            print("[Overlap] M3C2 analyzing done !")

            [os.remove(i) for i in glob.glob(self.workspace+"*_thin.laz")]
            [os.remove(i) for i in glob.glob(self.workspace+"*_clean.laz")]
        else:
            raise OSError("Pre-processing your data before")

    def Comparison(self,lineA,lineB,corePts,outName):
        query= tools.cloudcompare.open_file(self.CCoptions, [self.workspace + lineA, self.workspace + lineB, self.workspace + corePts])
        tools.cloudcompare.m3c2(query, self.workspace + self.m3c2File)
        tools.cloudcompare.last_file(self.workspace + corePts[0:-4] + "_20*.laz", outName)
        new_file= tools.cloudcompare.last_file(self.workspace + lineA[0:-4] + "_20*.laz")
        os.remove(new_file)
        new_file= tools.cloudcompare.last_file(self.workspace + lineA[0:-4] + "_M3C2_20*.laz")
        os.remove(new_file)
        new_file= tools.cloudcompare.last_file(self.workspace + lineB[0:-4] + "_20*.laz")
        os.remove(new_file)

    def WritingFile(self,filepath):
        inData= tools.lastools.read(filepath, extra_field=True)
        select1=inData["distance_uncertainty"]<self.filterDistUncertainty
        select2=np.logical_and(inData["m3c2_distance"]<self.m3c2Dist,inData["m3c2_distance"]>-self.m3c2Dist)
        select=np.logical_and(select1,select2)
        inData2= tools.lastools.filter_las(inData, select)
        extra=[(("m3c2_distance","float32"),inData2["m3c2_distance"]),
                (("distance_uncertainty","float32"),inData2["distance_uncertainty"])]

        tools.lastools.WriteLAS(filepath[0:-4] + "_clean.laz", inData2, extraField=extra)


workspace=r'G:\RENNES1\Loire_totale_automne2019\Loire_S01-01_S01-02_S02-01PART\04-QC\Recouvrement\ground_flightlines_C3'+'//'
params_file="m3c2_params2.txt"
projectName="Ardeche_18102021"
filtre_dist_uncertainty=0.5
filtre_m3c2Distance=10
param_openFile=['standard','LAS',"Loire49-1"]

rootsDict={100:[projectName+"_L","_C2_r_1.laz"]}

a=Overlap(workspace,params_file,projectName+"_C2_recouvrement.laz",
        [param_openFile,rootsDict,filtre_dist_uncertainty,filtre_m3c2Distance,2])
a.Preprocessing()
a.Processing()