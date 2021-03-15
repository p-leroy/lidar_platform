# -*- coding: utf-8 -*-
import numpy as np
import mmap,struct,os
from . import utils

class File(object):
    def __init__(self,filepath,mode="r"):
        self.filepath=filepath
        self._mode=mode
        self._Open()

    def _Open(self):
        if self._mode=="r":
            if not os.path.exists(self.filepath):
                raise OSError("No such file or directory: '%s'" %self.filepath)

            ext=os.path.splitext(self.filepath)[1]
            if ext==".sbf":
                listShiftPrec_temp=self._ComputeMetadata()
                self._reader=Reader(self.filepath+".data",self.metadata)
            elif ext==".data" and self.filepath[-9::]==".sbf.data":
                self._reader=Reader(self.filepath)
            else:
                raise OSError("Unrecognized file format : '%s'" %os.path.split(self.filepath)[1])

            self.header=self._reader.header
            self.points=self._reader.points
            self.scalarNames=self._reader.scalarNames
            self.listShiftPrecision=dict(zip(self.scalarNames[3::],listShiftPrec_temp))
        else:
            raise NotImplementedError("SBF files can only be opened in mode 'r' for now")

    def _ComputeMetadata(self):
        f=open(self.filepath,mode='r')
        tab=f.readlines()
        f.close()    
        temp=tab[2][12:-1].split(sep=",")
        self.metadata={"File_type":tab[0][0:-1],
                       "NbPoints":int(tab[1][7:-1]),
                       "GlobalShift":(float(temp[0]),float(temp[1]),float(temp[2])),
                       "NbScalarFields":int(tab[3][8:-1])}
        listShiftPrec_temp=[]
        for i in range(4,len(tab)):
            pos=tab[i].find("=")
            key=tab[i][0:pos]
            value=tab[i][(pos+1):-1]
            if value.find(",")==-1:
                listShiftPrec_temp+=[{}]
                self.metadata[key]=value
            else:
                valueSeg=value.split(sep=",")
                self.metadata[key]=valueSeg[0]
                s=valueSeg[1].split(sep="=")[1][0:-1]
                if len(valueSeg)>2:
                    p=valueSeg[2].split(sep="=")[1][0:-1]
                    listShiftPrecision+=[{"shift":float(s),"prec":float(p)}]
                else:
                    listShiftPrec_temp+=[{"shift":float(s)}]
        return listShiftPrec_temp

    
class Reader(object):
    def __init__(self,pathFileData,metadata={}):
        self._convention=utils.convention
        self.metadata=metadata
        if not os.path.exists(pathFileData):
            raise OSError("No such file or directory: '%s'" %pathFileData)

        f=open(pathFileData,'rb')
        self.__buffer=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        f.close()
        self._ComputeHeader()
        self._ComputePoints()
        self._ComputeSFnames()
        
    def _ComputeHeader(self):
        head_buff=self.__buffer[0:64]
        self.__data_size=len(self.__buffer[64::])
        head_tmp=struct.unpack('>2BQH3d',head_buff[0:36])
        self.header={'Key1':head_tmp[0],'Key2':head_tmp[1],'NbPoints':head_tmp[2],'NbScalarFields':head_tmp[3],
                     'InternalShift':(head_tmp[4],head_tmp[5],head_tmp[6]),'User_data':head_buff[36::]}
        self.NbSF=int(self.header['NbScalarFields']+3)
        if self.NbSF*4*self.header['NbPoints']!=self.__data_size:
            raise OSError("Unable to read data file :\n\tsize of each point and NbPoints don't match exactly file size !")

    def _ComputePoints(self):
        data_buff=self.__buffer[64::]
        liste_data=[]
        for i in range(0,self.header['NbPoints']):
            liste_data+=[struct.unpack('>'+str(self.NbSF)+'f',data_buff[i*self.NbSF*4:(i+1)*self.NbSF*4])]

        self.points=np.array(liste_data)
        self.points[:,0:3]+=self.header['InternalShift']

    def _ComputeSFnames(self):
        self.scalarNames=['X','Y','Z']
        if len(self.metadata)>0:
            liste_keys=list(self.metadata.keys())
            for i in range(4,len(liste_keys)):
                name=self.metadata[liste_keys[i]].lower()
                if name in self._convention.keys():
                    self.scalarNames+=[self._convention[name]]
                else:
                    self.scalarNames+=[self.metadata[liste_keys[i]].lower().replace('"','')]
        else:
            for i in range(0,self.header['NbScalarFields']):
                self.scalarNames+=["SF"+str(i+1)]


    
if __name__=="__main__":
    filepath=r'G:\RENNES1\BaptisteFeldmann\Developpements\extrait_1_v2.sbf'
    f=File(filepath)



        
            
        
            


            
