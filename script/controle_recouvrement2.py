# coding: utf-8
# Baptiste Feldmann
import numpy as np
import plateforme_lidar as PL
import matplotlib.pyplot as plt
import os,glob,copy,pickle
from joblib import Parallel,delayed
import importlib
importlib.reload(PL)

class Overlap(object):
    def __init__(self,workspace,m3c2File,waterSurfaceFile="",settings=[]):
        # settings=[['standard,'LAS','Loire49-2'],rootNames,distUncertainty=0.02,lineNbLength=2]
        self.workspace=workspace
        self.m3c2File=m3c2File
        self.waterSurface=waterSurfaceFile
        self.CCoptions=settings[0]
        self.rootNamesDict=settings[1]
        self.filterDistUncertainty=settings[2]
        self.lineNbLength=settings[3]
        self.keyList=list(self.rootNamesDict.keys())
        self.rootLength=len(self.rootNamesDict[self.keyList[0]][0])
        self._preprocessingStatus=False
    
    def _selectPairs(self,motif="*_thin.laz"):
        if os.path.exists(self.workspace+self.folder+"/comparison.pkl"):
            self.pairsDict=pickle.load(open(self.workspace+self.folder+"/comparison.pkl",'rb'))
        else:
            self.pairsDict=PL.calculs.select_pairs_overlap(self.workspace+self.folder+"/"+motif,[self.rootLength,self.lineNbLength])
            pickle.dump(self.pairsDict,open(self.workspace+self.folder+"/comparison.pkl",'wb'))
        
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

    def _filtering(self,workspace,inFile,outFile,filter_=[50,0.2]):
        data=PL.lastools.readLAS(workspace+inFile,extraField=True)
        select=np.logical_or(data["c2c_absolute_distances"]>filter_[0],data["c2c_absolute_distances_z"]>filter_[1])
        outData=PL.lastools.Filter_LAS(data,select)
        PL.lastools.writeLAS(workspace+outFile,outData)

    def Preprocessing(self,folder):
        print("[Overlap] Pre-processing...")
        self.folder=folder
        self.listFiles=[]
        self.listCompareLines=[]
        if self.folder=="C2":
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
                    
        elif self.folder=="C3":
            PL.cloudcompare.c2c_files(self.CCoptions,
                                      self.workspace+self.folder+"/",
                                      [os.path.split(i)[1] for i in glob.glob(self.workspace+self.folder+"/*_C3_r_thin.laz")],
                                      self.workspace+self.waterSurface,10,10)
            Parallel(n_jobs=20,verbose=1)(delayed(self._filtering)(self.workspace+self.folder+"/",i,i[0:-8]+"_1.laz") for i in [os.path.split(i)[1] for i in glob.glob(self.workspace+self.folder+"/*_C3_r_thin_C2C.laz")])
            for i in glob.glob(self.workspace+self.folder+"/*_C2C.laz"):
                os.remove(i)

            self._selectPairs(motif="*_thin_1.laz")
            if len(self.pairsDict.keys())==0:
                raise ValueError("Comparison dictionnary is empty")
            
            for i in self.pairsDict.keys():
                fileA=self._selectRootname(i)
                fileCorepts=fileA[0:-4]+"_thin_1.laz"
                for c in self.pairsDict[i]:
                    fileB=self._selectRootname(c)
                    fileResult=fileCorepts[0:-4]+"_m3c2_"+i+"and"+c+".laz"
                    self.listFiles+=[[fileA,fileB,fileCorepts,fileResult]]
                    self.listCompareLines+=[i+"_"+c]

        elif self.folder=="C2_C3":
            PL.cloudcompare.c2c_files(self.CCoptions,
                                        self.workspace+self.folder+"/",
                                        [os.path.split(i)[1] for i in glob.glob(self.workspace+self.folder+"/*_C2_r_thin.laz")],
                                        self.workspace+self.waterSurface,10,10)
            Parallel(n_jobs=20,verbose=1)(delayed(self._filtering)(self.workspace+self.folder+"/",i,i[0:-8]+"_1.laz") for i in [os.path.split(i)[1] for i in glob.glob(self.workspace+self.folder+"/*_C2_r_thin_C2C.laz")])
            for i in glob.glob(self.workspace+self.folder+"/*_C2C.laz"):
                os.remove(i)
            
            for i in [os.path.split(i)[1] for i in glob.glob(self.workspace+self.folder+"/*_C2_r.laz")]:
                fileA=i
                fileCorepts=fileA[0:-4]+"_thin_1.laz"
                fileB=fileA[0:-9]+"_C3_r.laz"
                fileResult=fileCorepts[0:-4]+"_m3c2_C2C3.laz"
                self.listFiles+=[[fileA,fileB,fileCorepts,fileResult]]
                self.listCompareLines+=[i[self.rootLength:self.rootLength+self.lineNbLength]]
        else:
            raise OSError("Unknown folder")
        
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
            self.results=Parallel(n_jobs=25,verbose=1)(delayed(self.AnalyzeFile)(self.workspace+self.folder+"/"+self.listFiles[i][-1],self.listCompareLines[i]) for i in range(0,len(self.listFiles)))
            np.savetxt(self.workspace+self.folder+"/save_results.txt",self.results,fmt='%s',delimiter=';',header='Comparaison;moyenne (m);ecart-type (m)')
            print("[Overlap] M3C2 analyzing done !")
        else:
            raise OSError("Pre-processing your data before")

    def Comparison(self,lineA,lineB,corePts,outName):
        query=PL.cloudcompare.open_file(self.CCoptions,[self.workspace+self.folder+"/"+lineA,self.workspace+self.folder+"/"+lineB,self.workspace+self.folder+"/"+corePts])
        PL.cloudcompare.m3c2(query,self.workspace+self.m3c2File)
        PL.cloudcompare.last_file(self.workspace+self.folder+"/"+corePts[0:-4]+"_20*.laz",outName)
        new_file=PL.cloudcompare.last_file(self.workspace+self.folder+"/"+lineA[0:-4]+"_20*.laz")
        os.remove(new_file)
        new_file=PL.cloudcompare.last_file(self.workspace+self.folder+"/"+lineA[0:-4]+"_M3C2_20*.laz")
        os.remove(new_file)
        new_file=PL.cloudcompare.last_file(self.workspace+self.folder+"/"+lineB[0:-4]+"_20*.laz")
        os.remove(new_file)

    def AnalyzeFile(self,filepath,compareID):
        inData=PL.lastools.readLAS(filepath,extraField=True)
        inData2=PL.lastools.Filter_LAS(inData,np.logical_not(np.isnan(inData["distance_uncertainty"])))
        inData3=PL.lastools.Filter_LAS(inData2,inData2["distance_uncertainty"]<self.filterDistUncertainty)
        m3c2_dist=inData3['m3c2_distance'][np.logical_not(np.isnan(inData3['m3c2_distance']))]

        if len(m3c2_dist)>100:
            output=[compareID,np.round(np.mean(m3c2_dist),3),np.round(np.std(m3c2_dist),3)]
        else:
            output=[compareID,"NotEnoughPoints","-"]
        return output


workspace=r'G:\RENNES1\Loire_totale_automne2019\Loire_Sully-sur-Loire_Checy\04-QC\Recouvrement'+'//'
params_file="m3c2_params.txt"
surface_water="C2_ground_thin_1m_watersurface_smooth5.laz"
filtre_dist_uncertainty=0.1
param_openFile=['standard','LAS',"Loire45-3"]

folder="C3"
rootsDict={40:["Sully-sur-Loire-Checy_M01_L","_C3_r.laz"],
            44:["Sully-sur-Loire-Checy_M02_L","_C3_r.laz"],
            99:["Sully-sur-Loire-Checy_M03_L","_C3_r.laz"]}
# rootsDict={85:["Loire-Briare-Sully-sur-Loire_M01-Briare-Sully-sur-Loire_L","_C3_r.laz"]}

a=Overlap(workspace,params_file,surface_water,[param_openFile,rootsDict,filtre_dist_uncertainty,2])
a.Preprocessing(folder)
a.Processing()

# #---Graphique---#
# list_filepath=glob.glob(workspace+folder+"/*_m3c2_*.laz")

# def func(filepath,distance_filter):
#     data=PL.lastools.readLAS_laspy(filepath,extraField=True)
#     subData1=PL.lastools.Filter_LAS(data,np.logical_not(np.isnan(data["distance_uncertainty"])))
#     del data
#     subData2=PL.lastools.Filter_LAS(subData1,subData1["distance_uncertainty"]<distance_filter)
    
#     m3c2_dist=subData2['m3c2_distance'][np.logical_not(np.isnan(subData2['m3c2_distance']))]
#     select=np.abs(m3c2_dist)<1
#     m3c2_dist=m3c2_dist[select]
        
#     if len(m3c2_dist)>100:
#         line_select=np.unique(np.random.randint(0,len(m3c2_dist),int(0.5*len(m3c2_dist))))
#         resultats=m3c2_dist[line_select]
#     else:
#         resultats=[]
#     return resultats


# #result=Parallel(n_jobs=20, verbose=2)(delayed(func)(i,filtre_dist_uncertainty) for i in list_filepath)
# #np.savez_compressed(workspace+folder+"/save_results_data_v1.npz",np.concatenate(result))


# f=np.load(workspace+folder+"/save_results_data_v1.npz")
# tab=f[f.files[0]]
# f.close()

# print(np.mean(tab))
# print(np.std(tab))

# plt.figure(1)
# plt.xlabel("Distance M3C2 (en cm)")
# plt.ylabel("Fréquence")
# plt.title('Histogramme des écarts en altitude\npour les données du canal vert')
# plt.hist(tab*100,bins=50,range=(-15,15),edgecolor='white')
# plt.ticklabel_format(axis="y",style='sci',scilimits=(0,0))
# #plt.text(x=-30,y=3000,s="Moyenne : -9cm\nEcart-type : 5.5cm")
# plt.savefig(workspace+folder+"/figure_C3_v1.png",dpi=150)
# #plt.show()
# #====================#




