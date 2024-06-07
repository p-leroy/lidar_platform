# coding: utf-8
# Baptiste Feldmann
import plateforme_lidar as PL
import shutil,os,glob
from joblib import Parallel,delayed

import importlib
importlib.reload(PL)

class Deliverable(object):
    def __init__(self,workspace,resolution,rootName):
        self.workspace=workspace
        self.rootName=rootName
        self.pixelSize=resolution
        if self.pixelSize<1:
            self.pixelSizeName=str(int(self.pixelSize*100))+"cm"
        else:
            self.pixelSizeName=str(int(self.pixelSize))+"m"
        self.channelSettings={"C2":["C2"],"C3":["C3"],"C2C3":["C2","C3"]}
        self.MKPsettings={"ground":["bathy","ground"],"nonground":["vegetation","building"]}

    def _clean(self):
        [os.remove(i) for i in glob.glob(self.workspace+self.rasterDir+"/*.laz")]
        [os.remove(i) for i in glob.glob(self.workspace+self.rasterDir+"/*.lax")]
        listDir=os.listdir(self.workspace+self.rasterDir+"/dalles")
        [shutil.move(self.workspace+self.rasterDir+"/dalles/"+i,self.workspace+self.rasterDir+"/"+i) for i in listDir]
        shutil.rmtree(self.workspace+self.rasterDir+"/dalles")

    def DTM(self,channel):
        self.rasterDir="_".join(["MNT",channel,self.pixelSizeName])
        os.mkdir(self.workspace+self.rasterDir)
        if "C2" in channel:
            query=f'las2las -i {self.workspace}LAS/C2/*.laz -keep_class 2 -cores 50 -odir {self.workspace+self.rasterDir} -olaz'
            utils.run(query)
        if "C3" in channel:
            query=f'las2las -i {self.workspace}LAS/C3/*.laz -keep_class 2 10 16 -cores 50 -odir {self.workspace+self.rasterDir} -olaz'
            utils.run(query)

        outName=[self.rootName,"MNT",self.pixelSizeName+".tif"]
        os.mkdir(self.workspace+self.rasterDir+"/dalles")
        utils.run(f'lasindex -i {self.workspace + self.rasterDir}/*.laz -cores 50')
        utils.run(f'lastile -i {self.workspace + self.rasterDir}/*.laz -tile_size 1000 -buffer 250 -cores 45 -odir {self.workspace + self.rasterDir}/dalles -o {self.workspace + self.rasterDir}/dalles/{self.rootName}_MNT.laz')
        utils.run("blast2dem -i " + self.workspace + self.rasterDir + "/dalles/*.laz -step " + str(self.pixelSize) + " -kill 250 -use_tile_bb -epsg 2154 -cores 50 -otif")
        tools.gdal.merge(glob.glob(self.workspace + self.rasterDir + "/dalles/*.tif"), self.workspace + self.rasterDir + "/dalles/" + "_".join(outName))
        self._clean()

    def DTM_bathy(self):
        self.rasterDir="_".join(["MNT_bathy_C3",self.pixelSizeName])
        os.mkdir(self.workspace+self.rasterDir)
        
        utils.run("las2las -i " + self.workspace + "LAS/C3/*.laz -keep_class 10 16 -cores 50 -odir " + self.workspace + self.rasterDir + " -olaz")

        outName=[self.rootName,"MNT",self.pixelSizeName+".tif"]
        os.mkdir(self.workspace+self.rasterDir+"/dalles")
        utils.run("lasindex -i " + self.workspace + self.rasterDir + "/*.laz -cores 50")
        utils.run("lastile -i " + self.workspace + self.rasterDir + "/*.laz -tile_size 1000 -buffer 250 -cores 45 -odir " + self.workspace + self.rasterDir + "/dalles -o " + self.workspace + self.rasterDir + "/dalles/" + self.rootName + "_MNT.laz")
        utils.run("blast2dem -i " + self.workspace + self.rasterDir + "/dalles/*.laz -step " + str(self.pixelSize) + " -kill 250 -use_tile_bb -epsg 2154 -cores 50 -otif")
        tools.gdal.merge(glob.glob(self.workspace + self.rasterDir + "/dalles/*.tif"), self.workspace + self.rasterDir + "/dalles/" + "_".join(outName))
        self._clean()

    def DSM(self,channel,opt="vegetation"):
        '''
        opt for MNS : "vegetation" or "vegetation_building"
        '''
        self.rasterDir="_".join(["MNS",opt,channel,self.pixelSizeName])
        os.mkdir(self.workspace+self.rasterDir)
        outName=[self.rootName,"MNS",self.pixelSizeName+".tif"]

        if "C2" in channel:
            if opt=="vegetation":
                utils.run("las2las -i " + self.workspace + "LAS/C2/*.laz -keep_class 2 5 -cores 50 -odir " + self.workspace + self.rasterDir + " -olaz")
            else:
                utils.run("las2las -i " + self.workspace + "LAS/C2/*.laz -keep_class 2 5 6 -cores 50 -odir " + self.workspace + self.rasterDir + " -olaz")
        if "C3" in channel:
            if opt=="vegetation":
                utils.run("las2las -i " + self.workspace + "LAS/C3/*.laz -keep_class 2 5 10 16 -cores 50 -odir " + self.workspace + self.rasterDir + " -olaz")
            else:
                utils.run("las2las -i " + self.workspace + "LAS/C3/*.laz -keep_class 2 5 6 10 16 -cores 50 -odir " + self.workspace + self.rasterDir + " -olaz")

        os.mkdir(self.workspace+self.rasterDir+"/dalles")
        os.mkdir(self.workspace+self.rasterDir+"/dalles/thin")

        utils.run("lasindex -i " + self.workspace + self.rasterDir + "/*.laz -cores 50")
        utils.run("lastile -i " + self.workspace + self.rasterDir + "/*.laz -tile_size 1000 -buffer 250 -cores 45 -odir " + self.workspace + self.rasterDir + "/dalles -o " + self.workspace + self.rasterDir + "/dalles/" + self.rootName + "_MNS.laz")
        utils.run("lasthin -i " + self.workspace + self.rasterDir + "/dalles/*.laz -step 0.2 -highest -cores 50 -odir " + self.workspace + self.rasterDir + "/dalles/thin -olaz")
        utils.run("blast2dem -i " + self.workspace + self.rasterDir + "/dalles/thin/*.laz -step " + str(self.pixelSize) + " -kill 250 -use_tile_bb -epsg 2154 -cores 50 -otif")
        tools.gdal.merge(glob.glob(self.workspace + self.rasterDir + "/dalles/thin/*.tif"), self.workspace + self.rasterDir + "/dalles/thin/" + "_".join(outName))
        self._clean()

    def DCM(self,channel):
        self.rasterDir="_".join(["MNC",channel,self.pixelSizeName])
        os.mkdir(self.workspace+self.rasterDir)
        MNCpath=self.workspace+self.rasterDir+"/"
        MNSpath=self.workspace+"MNS_vegetation_"+channel+"_"+self.pixelSizeName+"/thin/"
        MNTpath=self.workspace+"MNT_"+channel+"_"+self.pixelSizeName+"/"
        if not (os.path.exists(MNSpath+self.rootName+"_MNS_"+self.pixelSizeName+".tif") and os.path.exists(MNTpath+self.rootName+"_MNT_"+self.pixelSizeName+".tif")):
            raise Exception("MNS_vegetation or MNT aren't already computed !")

        outName=[self.rootName,"MNC",self.pixelSizeName+".tif"]
        listMNS=[os.path.split(i)[1] for i in glob.glob(MNSpath+"*00.tif")]
        listMNT=[]
        listMNC=[]
        for i in listMNS:
            splitCoords=i.split(sep="_")[-2::]
            listMNT+=[self.rootName+"_MNT_"+"_".join(splitCoords)]
            listMNC+=[self.rootName+"_MNC_"+"_".join(splitCoords)]
        Parallel(n_jobs=50,verbose=2)(delayed(tools.gdal.raster_calc)("((A-B)<0)*0+((A-B)>=0)*(A-B)", MNCpath + listMNC[i], MNSpath + listMNS[i], MNTpath + listMNT[i]) for i in range(0, len(listMNS)))
        tools.gdal.merge(glob.glob(MNCpath + "*.tif"), MNCpath + "_".join(outName))

    def Density(self,channel):
        #not finish
        DTM_path=self.workspace+"_".join(["MNT",channel,self.pixelSizeName])+"/"
        if not os.path.exists(DTM_path+self.rootName+"_MNT_"+self.pixelSizeName+".tif"):
            raise Exception("MNT isn't already computed !")

        outName=[self.rootName,"MNT","density",self.pixelSizeName+".tif"]
        os.mkdir(DTM_path+"density")
        os.mkdir(DTM_path+"density/final")

        utils.run("lasgrid -i " + DTM_path + "*.laz -step " + str(self.pixelSize) + " -use_tile_bb -counter_16bit -drop_class 10 -cores 50 -epsg 2154 -odir " + DTM_path + "density -odix _density -otif")
        listMNT=[os.path.split(i)[1] for i in glob.glob(DTM_path+"*00.tif")]
        Parallel(n_jobs=50,verbose=2)(delayed(tools.gdal.hole_filling)(DTM_path + "density/" + i[0:-4] + "_density.tif", DTM_path + i) for i in listMNT)
        tools.gdal.merge(glob.glob(DTM_path + "density/final/*.tif"), DTM_path + "density/final/" + "_".join(outName))
        
        [os.remove(i) for i in glob.glob(DTM_path+"density/*_density.tif")]
        listDir=os.listdir(DTM_path+"density/final")
        [shutil.move(DTM_path+"density/final/"+i,DTM_path+"density/"+i) for i in listDir]
        shutil.rmtree(DTM_path+"density/final")
    
    def MKP(self,channel,mode,settings=[]):
        '''
        mode : ground or nonground
        settings : ground=[vertical,horiz], nonground=[thinning step]
        '''
        #not finish
        self.rasterDir="MKP_"+mode
        outName=[self.rootName,"MKP",channel,mode+".laz"]
        os.mkdir(self.workspace+self.rasterDir)
        os.mkdir(self.workspace+self.rasterDir+"/dalles")
        params_dict={"ground":["2 16","-step 5 -adaptive "+str(settings[0])+" "+str(settings[-1])],
                    "nonground":["5 6","-step "+str(settings[0])+"-random"]}

        if "C2" in channel:
            utils.run("las2las -i " + self.workspace + "LAS/C2/*.laz -keep_class " + params_dict[mode][0] + " -cores 50 -odir " + self.workspace + self.rasterDir + " -olaz")

        if "C3" in channel:
            utils.run("las2las -i " + self.workspace + "LAS/C3/*.laz -keep_class " + params_dict[mode][0] + " -cores 50 -odir " + self.workspace + self.rasterDir + " -olaz")

        utils.run("lasindex -i " + self.workspace + self.rasterDir + "/*.laz -cores 50")
        utils.run("lastile -i " + self.workspace + self.rasterDir + "/*.laz -tile_size 1000 -buffer 25 -cores 45 -odir " + self.workspace + self.rasterDir + "/dalles -o MKP.laz")
        utils.run("lasthin -i " + self.workspace + self.rasterDir + "/dalles/*.laz " + params_dict[mode][1] + " -cores 50 -odix _thin -olaz")
        utils.run("lastile -i " + self.workspace + self.rasterDir + "/dalles/*_thin.laz -remove_buffer -cores 50 -olaz")
        utils.run("lasmerge -i " + self.workspace + self.rasterDir + "/dalles/*_thin_1.laz -o " + self.workspace + "_".join(outName))
        shutil.rmtree(self.workspace+self.rasterDir)

        
workspace=r'G:\RENNES1\Moselle_23092021\05-Traitements\Raster'+'//'

a=Deliverable(workspace,0.5,"Moselle_23092021")
#a.MKP("C2C3","ground",[0.1,1])
#a.MKP("C2C3","nonground",[0.5])
a.DTM("C2C3")
#a.DTM_bathy()
a.Density("C2C3")
#a.DSM("C2C3",'vegetation_building')
#a.DSM("C2C3",'vegetation')
#a.DCM("C2C3")




    
