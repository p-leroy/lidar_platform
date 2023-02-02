# -*- coding: utf-8 -*-

import mmap
import os
import struct
import time

import numpy as np

from . import misc


# ---PySBF---#
class PointCloud(object):
    """LAS data object

    Attributes:
        metadata (dict): {'vlrs': dict (info about LAS vlrs),'extraField': list (list of additional fields)}
        XYZ (numpy.ndarray): coordinates
        various attr (numpy.ndarray):

    Functionality:
        len('plateforme_lidar.tools.las_fmt'): number of points
        print('plateforme_lidar.tools.las_fmt'): list of attributes
        get attribute: lasdata.attribute or lasdata[attribute]
        set attribute: lasdata.attribute=value or lasdata[attribute]=value
        create attribute: setattr(lasdata,attribute,value) or lasdata[attribute]=value
    """

    def __len__(self):
        return len(self.XYZ)

    def __str__(self):
        return "\n".join(self.__dict__.keys())

    def __repr__(self):
        var = len(self.metadata["ScalarNames"])
        return f'<SBF object of {len(self.XYZ)} points with {var} attributes>'

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    pass


def readMetadataFile(filepath):
    f = open(filepath,mode='r')
    lines = []
    for i in f.readlines():
        lines += [i.replace("\n", "")]
    f.close()    

    temp = lines[2].split("=")[1]
    metadata = {"NbPoints":int(lines[1].split("=")[1]),
                "GlobalShift":tuple(float(i) for i in temp.split(",")),
                "NbScalarFields":int(lines[3].split('=')[1])}

    listShiftPrec = {}
    listScalarNames = []
    for line in lines[4::]:
        pos = line.find("=")
        value = line[(pos + 1)::]
        if value.find(",") == -1:
            listScalarNames += [misc.camel_to_snake(value)]
        else:
            valueSplit = value.split(sep=",")
            listScalarNames += [misc.camel_to_snake(valueSplit[0])]
            s = valueSplit[1].split(sep="=")[1][0: -1]
            listShiftPrec[misc.camel_to_snake(valueSplit[0])] = {"shift": float(s)}
            if len(valueSplit)>2:
                p=valueSplit[2].split(sep="=")[1][0: -1]
                listShiftPrec[misc.camel_to_snake(valueSplit[0])]["prec"] = float(p)
    metadata["ScalarNames"] = listScalarNames
    metadata["ShiftPrecision"] = listShiftPrec
    return metadata


def read(filepath):
    output = PointCloud()

    ext = os.path.splitext(filepath)[1]
    if ext == ".sbf":
        output['metadata'] = readMetadataFile(filepath)
        reader = Reader(filepath + ".data", output.metadata)
    elif ext == ".data" and filepath[-9::] == ".sbf.data":
        reader = Reader(filepath)
        output['metadata']={'NbPoints': reader.header['NbPoints'],
                            'GlobalShift': (0.0,0.0,0.0),
                            'NbScalarFields': reader.header['NbScalarFields'],
                            'ScalarNames': [f'SF{i}' for i in range(1,reader.header['NbScalarFields']+1)],
                            'ShiftPrecision':{}}
    else:
        raise Exception("Unrecognized file format : '%s'" %os.path.split(filepath)[1])

    output['XYZ'] = np.array([np.copy(reader.points['X']),
                              np.copy(reader.points['Y']),
                              np.copy(reader.points['Z'])]).transpose()
    output.XYZ += reader.header['InternalShift']
    for i in output.metadata['ScalarNames']:
        output[i] = np.copy(reader.points[i])
    return output


class Reader(object):
    def __init__(self,pathFileData,metadata={}):
        self.metadata=metadata
        if not os.path.exists(pathFileData):
            raise OSError("No such file or directory: '%s'" %pathFileData)

        f=open(pathFileData,'rb')
        self._buffer=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        f.close()
        self.ComputeHeader()
        self.ComputePoints()
        
    def ComputeHeader(self):
        head_buff=self._buffer[0:64]
        self._data_size=len(self._buffer[64::])
        head_tmp=struct.unpack('>2BQH3d',head_buff[0:36])
        self.header={'Key1':head_tmp[0],'Key2':head_tmp[1],'NbPoints':head_tmp[2],'NbScalarFields':head_tmp[3],
                     'InternalShift':(head_tmp[4],head_tmp[5],head_tmp[6]),'User_data':head_buff[36::]}
        self.NbSF=int(self.header['NbScalarFields']+3)
        if self.NbSF*4*self.header['NbPoints']!=self._data_size:
            raise OSError("Unable to read data file :\n\tsize of each point and NbPoints don't match exactly file size !")

    def ComputePoints(self):
        try:
            temp=list(zip(['X','Y','Z']+self.metadata['ScalarNames'],['>f']*(len(self.metadata['ScalarNames'])+3)))
        except:
            temp=list(zip(['X','Y','Z']+[f'SF{i}' for i in range(1,self.NbSF-2)],['>f']*self.NbSF))
        
        self.points=np.frombuffer(self._buffer,dtype=np.dtype(temp),count=self.header['NbPoints'],offset=64)


class Write(object):
    def __init__(self,filepath,data):
        print("[Writing LAS file]..",end="")
        self._start=time.time()
        self.filepath=filepath
        self.data_length=len(data)
        ext=os.path.splitext(filepath)[1]
        if ext==".sbf":
            self.writeMetadataFile(data.metadata)
            self.output=open(self.filepath+".data",mode='wb')
        elif ext==".data" and filepath[-9::]==".sbf.data":
            self.output=open(self.filepath,mode='wb')
        else:
            raise Exception("Unrecognized file format : '%s'" %os.path.split(filepath)[1])

        self.internalShift=np.min(data.XYZ,axis=0)
        self.writeHeader(data)
        self.writePoints(data)
        self.output.close()
        print("done")
    
    def __repr__(self):
        return "Write "+str(len(self.data_length))+" points in "+str(round(time.time()-self._start,1))+" sec"

    def writeMetadataFile(self,metadata):
        lines = ["[SBF]"]
        lines += [f'Points={metadata["NbPoints"]}',
                  "GlobalShift=" + str(metadata["GlobalShift"])[1 : -1],
                  f'SFCount={metadata["NbScalarFields"]}']
        for i in range(1,metadata["NbScalarFields"] + 1):
            name = metadata["ScalarNames"][i - 1]
            line = f'SF{i} = {misc.snake_to_camel(name)}'
            if name in metadata["ShiftPrecision"].keys():
                for key in metadata["ShiftPrecision"][name].keys():
                    line += f', "{key[0:1]}={metadata["ShiftPrecision"][name][key]}"'
            lines += [line]

        with open(self.filepath,mode='w') as f:
            f.write("\n".join(lines))

    def writeHeader(self,pointcloud):
        header = [42,
                  42,
                  self.data_length,pointcloud.metadata["NbScalarFields"],
                  self.internalShift[0],
                  self.internalShift[1],
                  self.internalShift[2],
                  b'']
        header_bytes = struct.pack('>2BQH3d28s', *header)
        if len(header_bytes) != 64:
            raise Exception(f'Header issue, must be 64 bytes and got {len(header_bytes)} bytes')

        self.output.write(header_bytes)
    
    def writePoints(self,pointcloud):
        listNames=['X','Y','Z']+pointcloud.metadata['ScalarNames']
        listFormat=['>f']*(pointcloud.metadata['NbScalarFields']+3)
        dt=np.dtype(list(zip(listNames,listFormat)))

        temp=[pointcloud.XYZ-self.internalShift]
        for i in pointcloud.metadata["ScalarNames"]:
            temp+=[np.reshape(getattr(pointcloud,i),(-1,1))]
        temp=np.hstack(temp)
        tab=[]
        for i in temp:
            tab+=[tuple(i)]
        points=np.array(tab,dtype=dt)
        self.output.write(points.tobytes())
