# coding: utf-8
# Baptiste Feldmann
import numpy as np
import glob,os,mmap,struct,copy,time,warnings
from . import utils
import pylas,laspy

def Filter_LAS(obj,select):
    """Filtering lasdata

    Args:
        obj ('plateforme_lidar.utils.lasdata'): lasdata object
        select (list or int): list of boolean, list of integer or integer

    Returns:
        'plateforme_lidar.utils.lasdata': filtering lasdata object
    """
    if type(select)==list or type(select)==np.ndarray:
        if not len(select)==len(obj):
            select=np.array(select)[np.argsort(select)]

    obj_new=utils.lasdata()
    obj_new['metadata']=obj.metadata
    for i in obj.__dict__.keys():
        if i != 'metadata':
            setattr(obj_new,i,getattr(obj,i)[select])
    return obj_new

def Merge_LAS(listObj):
    """Merge lasdata

    Args:
        listObj (list): list of 'plateforme_lidar.utils.lasdata' type

    Returns:
        'plateforme_lidar.utils.lasdata': lasdata merged
    """
    merge=utils.lasdata()
    merge['metadata']=listObj[0].metadata
    merge['metadata']['extraField']=[]

    for feature in listObj[0].__dict__.keys():
        if feature not in ['metadata']:
            merge[feature]=np.concatenate([i[feature] for i in listObj],axis=0)
    return merge

def Filter_WDP(lines,select):
    """Filtering WDP tab

    Args:
        lines (list): list of waveforms
        select (list): list of boolean or list of index

    Returns:
        list : list of extracted waveforms
    """
    if len(select)==len(lines):
        select=np.where(select)[0]
    else:
        select=np.array(select)[np.argsort(select)]
    
    return [lines[i] for i in select]

def Update_ByteOffset(lasobj,waveforms,byte_offset_start=60):
    """Updating byte offset to waveform data

    Args:
        lasobj ('plateforme_lidar.utils.lasdata'): LAS dataset to update
        waveforms (list): list of waveforms
        byte_offset_start (int, optional): byte number of first line in WDP file. Defaults to 60.

    Raises:
        ValueError: Las file must have same number of points than waveforms !
    """
    newOffset=[byte_offset_start]
    sizes=np.uint16(lasobj.waveform_packet_size)
    if len(lasobj)!=len(waveforms):
        raise ValueError("Las file must have same number of points than waveforms !")

    for i in range(0,len(lasobj)):
        newOffset+=[np.uint64(newOffset[i]+sizes[i])]
    lasobj.byte_offset_to_waveform_data=newOffset[0:-1]

def read_VLRbody(vlrs):
    # reading VLR in LAS file with LasPy
    # Can only read waveform, bbox tile and projection vlrs
    liste={}
    for vlr in vlrs:
        if vlr.record_id>=100 and vlr.record_id<=356:
            #read waveform vlrs
            # (Bits/sample,wavefm compression type,nbr of samples,Temporal spacing,digitizer gain,digitizer offset)
            liste[vlr.record_id]=struct.unpack("=BBLLdd",vlr.VLR_body)
        elif vlr.record_id==10:
            #read bbox tile vlrs :
            # (level,index,implicit_lvl,reversible,buffer,min_x,max_x,min_y,max_y)
            liste[vlr.record_id]=struct.unpack("=2IH2?4f",vlr.VLR_body)
        elif vlr.record_id==34735:
            #read Projection
            # (KeyDirectoryVersion,KeyRevision,MinorRevision,NumberofKeys)+ n*(KeyId,TIFFTagLocation,Count,Value_offset)
            listGeoKeys=struct.unpack("=4H",vlr.VLR_body[0:8])
            for i in range(0,int((len(vlr.VLR_body)-8)/8)):
                    temp=struct.unpack("=4H",vlr.VLR_body[8*(i+1):8*(i+1)+8])
                    if temp[1]==0 and temp[2]==1:
                        listGeoKeys+=temp
            liste[vlr.record_id]=listGeoKeys            
    return liste

def read_pylas_VLRbody(vlrs):
    # reading VLR in LAS file with PyLas
    # Can only read waveform, bbox tile and projection vlrs
    liste={}
    for vlr in vlrs:
        if vlr.record_id>=100 and vlr.record_id<=356:
            #read waveform vlrs
            # (Bits/sample,wavefm compression type,nbr of samples,Temporal spacing,digitizer gain,digitizer offset)
            liste[vlr.record_id]=struct.unpack("=BBLLdd",vlr.record_data_bytes())
        elif vlr.record_id==10:
            #read bbox tile vlrs :
            # (level,index,implicit_lvl,reversible,buffer,min_x,max_x,min_y,max_y)
            liste[vlr.record_id]=struct.unpack("=2IH2?4f",vlr.record_data_bytes())
        elif vlr.record_id==34735:
            #read Projection
            # (KeyDirectoryVersion,KeyRevision,MinorRevision,NumberofKeys)+ n*(KeyId,TIFFTagLocation,Count,Value_offset)
            listGeoKeys=struct.unpack("=4H",vlr.record_data_bytes()[0:8])
            for i in range(0,int((len(vlr.record_data_bytes())-8)/8)):
                    temp=struct.unpack("=4H",vlr.record_data_bytes()[8*(i+1):8*(i+1)+8])
                    if temp[1]==0 and temp[2]==1:
                        listGeoKeys+=temp
            liste[vlr.record_id]=listGeoKeys            
    return liste

def pack_VLRbody(dictio):
    # writing VLR in LAS file with LasPy
    # Can only write waveform, bbox tile and projection vlrs
    liste=[]
    size=0
    if len(dictio)>0:
        for i in dictio.keys():
            if i >=100 and i<=356 :
                temp=laspy.header.VLR(user_id="LASF_Spec",record_id=i,VLR_body=struct.pack("=BBLLdd",*dictio[i]))
            elif i==10:
                temp=laspy.header.VLR(user_id="LAStools",record_id=i,VLR_body=struct.pack("=2IH2?4f",*dictio[i]))
            elif i==34735:
                fmt="="+str(len(dictio[i]))+"H"
                temp=laspy.header.VLR(user_id="LASF_Projection",record_id=i,VLR_body=struct.pack(fmt,*dictio[i]))
            else:
                raise ValueError("VLR.record_id unknown : "+str(i))

            size+=len(temp)
            liste+=[temp]
    return liste,size

def VLRS_keys(vlrs,geokey):
    """Adding geokey in VLR Projection

    Args:
        vlrs (dict): lasdata.metadata['vlrs']
        geokey (dict): geokey={"Vertical":epsg,"Projected":epsg}

    Returns:
        dict: updated vlrs
    """
    vlrs_copy=vlrs.copy()
    if 34735 in vlrs.keys():
        vlrs_dict={}
        for i in range(0,int(len(vlrs[34735])/4)):
            num=i*4
            vlrs_dict[vlrs[34735][num]]=vlrs[34735][num+1:num+4]
    else:
        vlrs_dict=utils.geokey_standard

    for i in list(geokey.keys()):
        vlrs_dict[utils.CRS_key[i]]=[0,1,geokey[i]]

    vlrs_sort=np.sort(list(vlrs_dict.keys()))
    vlrs_final=[]
    for i in vlrs_sort:
        vlrs_final+=[i]
        vlrs_final+=vlrs_dict[i]
    vlrs_final[3]=len(vlrs_sort)-1
    vlrs_copy[34735]=tuple(vlrs_final)
    return vlrs_copy
       
class writeLAS(object):
    def __init__(self,filepath,data,format_id=1,extraField=[],waveforms=[]):
        """Writing LAS 1.3 (only) with LasPy

        Args:
            filepath (str): output file path (extensions= .las or .laz)
            data ('plateforme_lidar.utils.lasdata'): lasdata object
            format_id (int, optional): data format id according to ASPRS convention (standard mode=1, fwf mode=4). Defaults to 1.
            extraField (list, optional): list of additional fields [(("name","type format"),listData),...]
                ex: [(("depth","float32"),numpy.ndarray),(("value","uint8"),numpy.ndarray)]. Defaults to [].
            waveforms (list, optional): list of waveforms to save in external WDP file. Make sure that format_id is compatible with wave packet (ie. 4,5,9 or 10). Default to []
        """
        # standard : format_id=1 ; fwf : format_id=4
        print("[Writing LAS file]..",end="")
        self._start=time.time()
        if filepath[-4::]==".laz":
            compressed=True
            self.filepathTrue=filepath
            self.filepath=filepath[0:-4]+"_temp.las"
        else:
            compressed=False
            self.filepath=filepath
        self.output=data
        self.formatId=format_id
        self.LAS_fmt=utils.LAS_format()
        
        self.createHeader()
        if len(extraField)>0:
            self.writeExtraField(extraField)
        else:
            self.outFile=laspy.file.File(self.filepath,self.header,self.vlrs_list,'w')

        self.writeAttr()
        self.outFile.close()
        if compressed:
            self.do_compression()
        print("done !")

        if len(waveforms)>0 and self.formatId in [4,5,9,10]:
            self.waveDataPacket(waveforms)
    
    def __repr__(self):
        return "Write "+str(len(self.output))+" points in "+str(round(time.time()-self._start,1))+" sec"

    def waveDataPacket(self,waveforms):
        # write external waveforms in WDP file not compressed
        # Future improvement will make writing compressed possible
        nbrPoints=len(self.output)
        sizes=np.uint16(self.output.waveform_packet_size)
        offsets=np.uint64(self.output.byte_offset_to_waveform_data)
        pkt_desc_index=self.output.wave_packet_desc_index
        vlrs=self.output.metadata['vlrs']

        if not all(offsets[1::]==(offsets[0:-1]+sizes[0:-1])):
            raise ValueError("byte offset list is not continuous, re-compute your LAS dataset")
    
        pourcent=[int(i*nbrPoints) for i in [0.2,0.4,0.6,0.8,0.98]]
        print("[Writing waveform data packet] : %d waveforms" %len(waveforms))

        print("0%..",end="")
        with open(self.filepath[0:-4]+".wdp","wb") as wdpFile :
            wdpFile.write(utils.headerWDP_binary)
            for i in range(0,nbrPoints):
                if i in pourcent:
                    print("%d%%.." %(20*(pourcent.index(i)+1)),end='')
                
                if len(waveforms[i])!=(sizes[i]/2):
                    raise ValueError("Size of waveform n°"+str(i)+" is not the same in LAS file")
                
                try:
                    vlr_body=vlrs[pkt_desc_index[i]+99]
                except:
                    raise ValueError("Number of the wave packet desc index not in VLRS !")

                length=int(vlr_body[2])
        
                try:
                    test=struct.pack(str(length)+'h',*np.int16(waveforms[i]))
                    wdpFile.write(test)
                except :
                    raise ValueError(str(length))
        print("done !")

    def createHeader(self):
        #Create header from point cloud in LAS 1.3 only
        today=utils.date()
        scale=0.001
        header_size=235
        dataMin=np.min(self.output.XYZ,axis=0)
        dataMax=np.max(self.output.XYZ,axis=0)
        
        offsets=np.int64(dataMin/1000)*1000
        self.vlrs_list,vlrs_size=pack_VLRbody(self.output.metadata['vlrs'])
        pt_return_count=[0]*5
        unique,counts=np.unique(self.output.return_num,return_counts=True)
        for i in unique:
            try:
                pt_return_count[i-1]=counts[i-1]
            except: pass

        if self.formatId in [4,5,9,10]:
            globalEncoding=2
        else:
            globalEncoding=0
    
        features={'global_encoding': globalEncoding,'system_id':self.LAS_fmt.identifier["system_identifier"],"software_id":self.LAS_fmt.identifier["generating_software"],
                  'created_day':today.day,'created_year':today.year,'header_size':header_size,'data_offset':header_size+vlrs_size,'num_variable_len_recs':len(self.vlrs_list),
                  'data_record_length': self.LAS_fmt.dataRecordLen[self.formatId],'legacy_point_records_count': len(self.output),'legacy_point_return_count': pt_return_count,
                  'x_scale': scale, 'y_scale': scale, 'z_scale': scale,'x_offset': offsets[0],'y_offset': offsets[1], 'z_offset': offsets[2], 'x_max': dataMax[0], 'x_min': dataMin[0], 'y_max': dataMax[1],
                  'y_min': dataMin[1],'z_max': dataMax[2], 'z_min': dataMin[2], 'point_records_count': len(self.output),'point_return_count': pt_return_count}
        self.header=laspy.header.Header(1.3,self.formatId,features)

    def writeExtraField(self,extra_field):
        #writing additional fields
        vlr_bod=b''
        data_rec_len=0
        for i in extra_field:
            vlr_bod+=laspy.header.ExtraBytesStruct(name=i[0][0],data_type=self.LAS_fmt.fmtNameValue[i[0][1]]).to_byte_string()
            data_rec_len+=self.LAS_fmt.fmtNameSize[i[0][1]]

        extra_dim_vlr=laspy.header.VLR(user_id="LASF_Spec",
                                       record_id=4,
                                       description="Extras_fields",
                                       VLR_body=vlr_bod)
        self.header.data_record_length+=int(data_rec_len)
        
        self.outFile=laspy.file.File(self.filepath,self.header,self.vlrs_list+[extra_dim_vlr],'w')            
        for i in extra_field:
            self.outFile.writer.set_dimension(i[0][0],i[1])

    def writeAttr(self):
        #writing conventional fields
        coords_int=np.array((self.output.XYZ-self.outFile.header.offset)/self.outFile.header.scale,dtype=np.int32)
        self.outFile.X=coords_int[:,0]
        self.outFile.Y=coords_int[:,1]
        self.outFile.Z=coords_int[:,2]
        for i in self.LAS_fmt.recordFormat[self.formatId]:
            try:
                data=getattr(self.output,i[0])
                setattr(self.outFile,i[0],getattr(np,i[1])(data))
            except:
                warnings.warn("Warning: Not possible to write attribute : "+i[0])

    def do_compression(self):
        #compressed with laszip if file extension is .laz
        #LasPy can't write data directly compressed, firstly write in .las extension then compressed output file
        utils.Run_bis("laszip -i "+self.filepath+" -o "+self.filepathTrue)
        os.remove(self.filepath)
        self.filepath=self.filepathTrue

def readLAS_laspy(filepath,extraField=False):
    """Reading LAS with LasPy

    Args:
        filepath (str): input LAS file (extensions: .las or .laz)
        extraField (bool, optional): True if you want to load additional fields. Defaults to False.

    Returns:
        'plateforme_lidar.utils.lasdata': lasdata object
    """
    f=laspy.file.File(filepath,mode='r')
    LAS_fmt=utils.LAS_format()

    metadata={"vlrs":read_VLRbody(f.header.vlrs),"extraField":[]}
    output=utils.lasdata()
    
    for i in LAS_fmt.recordFormat[f.header.data_format_id]:
        try :
            output[i[0]]=np.array(getattr(f,i[0]))  
        except:
            print("[LasPy] "+str(i[0])+" not found !")  

    coords=np.array([f.X,f.Y,f.Z])
    scales=np.transpose(np.array([f.header.scale]))
    offsets=np.transpose(np.array([f.header.offset]))
    fields=coords*scales+offsets
    setattr(output,"XYZ",fields.transpose())

    if extraField:
        extra=f.reader.extra_dimensions
        for i in extra:
            name=i.get_name().decode('utf-8').replace('\x00',"").replace(' ','_')
            nameStd=name.replace('(','').replace(')','').lower()
            metadata["extraField"]+=[nameStd]
            output[nameStd]=np.copy(f.reader.get_dimension(name))

    output['metadata']=metadata
    f.close()
    return output

def readLAS(filepath,extraField=False):
    """Reading LAS with PyLas

    Args:
        filepath (str): input LAS file (extensions: .las or .laz)
        extraField (bool, optional): True if you want to load additional fields. Defaults to False.

    Returns:
        'plateforme_lidar.utils.lasdata': lasdata object
    """
    f=pylas.read(filepath)
    LAS_fmt=utils.LAS_format()
    
    metadata={"vlrs":read_pylas_VLRbody(f.vlrs),"extraField":[]}
    output=utils.lasdata()

    for i in LAS_fmt.recordFormat[f.header.point_format_id]:
        try:
            output[i[0]]=np.array(getattr(f,i[0]))
        except:
            print("[Pylas] "+str(i[0])+" not found !")      

    coords=np.array([f.X,f.Y,f.Z])
    scales=np.transpose(np.array([f.header.scales]))
    offsets=np.transpose(np.array([f.points.offsets]))
    fields=coords*scales+offsets
    output['XYZ']=fields.transpose()

    if extraField:
        for i in list(f.points.extra_dimensions_names):
            name=i.replace('(','').replace(')','').lower()
            metadata['extraField']+=[name]
            output[name]=f[i]

    output['metadata']=metadata       
    return output

def sortLASdata(data,names,mode='standard'):
    namesAttr=utils.fields_names[mode]
    data_sort=np.copy(data)
    names_sort=np.copy(names)
    for i in namesAttr:
        listNames=list(names_sort)
        idx_true=namesAttr.index(i)+3
        if i not in names_sort:
            raise ValueError("Attribute %s isn't present in your column names !" %i)
        if listNames.index(i)!=idx_true:
            data_sort[:,[listNames.index(i),idx_true]]=data_sort[:,[idx_true,listNames.index(i)]]
            names_sort[[listNames.index(i),idx_true]]=names_sort[[idx_true,listNames.index(i)]]
    return data_sort,names_sort

def readWDP(lasfile,lasdata):
    """Reading waveforms in WDP file

    Args:
        lasfile (str): path to LAS file
        lasdata ('plateforme_lidar.utils.lasdata'): lasdata object

    Raises:
        ValueError: if for one point wave data packet descriptor is not in VLRS

    Returns:
        list: list of waveform (length of each waveform can be different)
    """
    nbrPoints=len(lasdata)
    sizes=np.uint16(lasdata.waveform_packet_size)
    offset=np.uint64(lasdata.byte_offset_to_waveform_data)
    pkt_desc_index=lasdata.wave_packet_desc_index
    vlrs=lasdata.metadata['vlrs']
    
    pourcent=[int(i*nbrPoints) for i in [0.2,0.4,0.6,0.8,0.98]]
    print("[Reading waveform data packet] : %d waveforms" %nbrPoints)
    
    with open(lasfile[0:-4]+".wdp",'rb') as wdp:
        dataraw=mmap.mmap(wdp.fileno(),os.path.getsize(lasfile[0:-4]+".wdp"),access=mmap.ACCESS_READ)

    lines=[]
    print("0%..",end="")
    for i in range(0,nbrPoints):
        if i in pourcent:
            print("%d%%.." %(20*(pourcent.index(i)+1)),end='')

        try:
            vlr_body=vlrs[pkt_desc_index[i]+99]
        except:
            raise ValueError("Number of the wave packet desc index not in VLRS !")
        
        length=int(vlr_body[2])
        line=np.array(struct.unpack(str(length)+'h',dataraw[offset[i]:(offset[i]+sizes[i])]))        
        lines+=[np.round_(line*vlr_body[4]+vlr_body[5],decimals=2)]
    print("done !")
    return lines

def read_orthofwf(workspace,lasfile):
    print("[Read waveform data packet] : ",end='\r')
    f=laspy.file.File(workspace+lasfile)
    nbr_pts=int(f.header.count)
    pourcent=[int(0.2*nbr_pts),int(0.4*nbr_pts),int(0.6*nbr_pts),int(0.8*nbr_pts),int(0.95*nbr_pts)]
    try :
        sizes=np.int_(f.waveform_packet_size)
    except :
        sizes=np.int_(f.points['point']['wavefm_pkt_size'])
    
    offset=np.int_(f.byte_offset_to_waveform_data)
    
    wdp=open(workspace+lasfile[0:-4]+".wdp",'rb')
    data=mmap.mmap(wdp.fileno(),0,access=mmap.ACCESS_READ)
    temp=read_VLRbody(f.header.vlrs)
    if len(f.header.vlrs)==1:
        vlr_body=temp[list(temp.keys())[0]]
    else:
        vlr_body=temp[f.header.vlrs[f.wave_packet_desc_index[0]].record_id]
        
    anchor_z=f.z_t*f.return_point_waveform_loc
    step_z=f.z_t[0]*vlr_body[3]
    lines=[]
    length=int(vlr_body[2])
    prof=[np.round_(anchor_z-(step_z*c),decimals=2) for c in range(0,length)]
    prof=np.transpose(np.reshape(prof,np.shape(prof)))
    for i in range(0,nbr_pts):
        if i in pourcent:
            print("%d%%-" %(25+25*pourcent.index(i)),end='\r')

        line=np.array(struct.unpack(str(length)+'h',data[offset[i]:offset[i]+sizes[i]]))
        lines+=[np.round_(line*vlr_body[4]+vlr_body[5],decimals=2)]
    
    wdp.close()
    f.close()
    print("done !")
    return np.stack([lines,prof]),vlr_body[3],np.round_(step_z,decimals=2)
