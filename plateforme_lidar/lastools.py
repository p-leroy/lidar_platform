# coding: utf-8
# Baptiste Feldmann
import numpy as np
import os,mmap,struct,copy,time,warnings
from . import utils
import laspy
from datetime import datetime,timezone

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
    listFeatures=list(obj.__dict__.keys())
    listFeatures.remove("metadata")

    for i in listFeatures:
        setattr(obj_new,i,getattr(obj,i)[select])
    return obj_new

def Merge_LAS(listObj):
    """Merge lasdata
    The returned structure takes format of the first in the list
    All the extraFields aren't kept

    Args:
        listObj (list): list of 'plateforme_lidar.utils.lasdata' type

    Returns:
        'plateforme_lidar.utils.lasdata': lasdata merged
    """
    merge=utils.lasdata()
    merge['metadata']=copy.deepcopy(listObj[0].metadata)
    merge['metadata']['extraField']=[]
    listFeatures=list(listObj[0].__dict__.keys())
    [listFeatures.remove(i) for i in listObj[0].metadata["extraField"]+["metadata"]]
    
    for feature in listFeatures:
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
    sizes=np.uint16(lasobj.wavepacket_size)
    if len(lasobj)!=len(waveforms):
        raise ValueError("Las file must have same number of points than waveforms !")

    for i in range(0,len(lasobj)):
        newOffset+=[np.uint64(newOffset[i]+sizes[i])]
    lasobj.wavepacket_offset=newOffset[0:-1]

def read_VLRbody(vlrs):
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
        vlrs_dict=utils.GEOKEY_STANDARD

    for i in list(geokey.keys()):
        vlrs_dict[utils.CRS_KEY[i]]=[0,1,geokey[i]]

    vlrs_sort=np.sort(list(vlrs_dict.keys()))
    vlrs_final=[]
    for i in vlrs_sort:
        vlrs_final+=[i]
        vlrs_final+=vlrs_dict[i]
    vlrs_final[3]=len(vlrs_sort)-1
    vlrs_copy[34735]=tuple(vlrs_final)
    return vlrs_copy
       
class writeLAS(object):
    def __init__(self,filepath,data,format_id=1,extraFields=[],waveforms=[],parallel=True):
        """Writing LAS 1.3 with LasPy

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
        self.output_data=data
        del data
        self.LAS_fmt=utils.LAS_FORMAT()
        # new_header=self.createHeader("1.3",format_id)
        # pointFormat=laspy.PointFormat(format_id)
        # for extraField in extraFields:
        #     pointFormat.add_extra_dimension(laspy.ExtraBytesParams(name=extraField["name"],type=extraField["type"],description="Extras_fields"))
        # new_points=laspy.PackedPointRecord(points,point_format=pointFormat)

        self.point_record=laspy.LasData(header=self.createHeader("1.3",format_id),points=laspy.ScaleAwarePointRecord.zeros(len(self.output_data),header=self.createHeader("1.3",format_id)))

        for extraField in extraFields:
            self.point_record.add_extra_dim(laspy.ExtraBytesParams(name=extraField["name"],type=getattr(np,extraField["type"]),description="Extras_fields"))
            setattr(self.point_record,extraField["name"],extraField["data"])

        self.writeAttr()
        self.point_record.write(filepath,laz_backend=utils.LASPY_PARALLEL_BACKEND[parallel])
        print("done !")

        if len(waveforms)>0 and format_id in [4,5,9,10]:
            self.waveDataPacket(filepath,waveforms)
    
    def __repr__(self):
        return "Write "+str(len(self.output_data))+" points in "+str(round(time.time()-self._start,1))+" sec"

    def waveDataPacket(self,filepath,waveforms):
        # write external waveforms in WDP file not compressed
        # Future improvement will make writing compressed possible
        nbrPoints=len(self.output_data)
        sizes=np.uint16(self.output_data.wavepacket_size)
        offsets=np.uint64(self.output_data.wavepacket_offset)
        pkt_desc_index=self.output_data.wavepacket_index
        vlrs=self.output_data.metadata['vlrs']

        if not all(offsets[1::]==(offsets[0:-1]+sizes[0:-1])):
            raise ValueError("byte offset list is not continuous, re-compute your LAS dataset")
    
        pourcent=[int(i*nbrPoints) for i in [0.2,0.4,0.6,0.8,0.98]]
        print("[Writing waveform data packet] : %d waveforms" %len(waveforms))

        print("0%..",end="")
        with open(filepath[0:-4]+".wdp","wb") as wdpFile :
            wdpFile.write(utils.HEADER_WDP_BYTE)
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

    def createHeader(self,version,formatId):
        #Create header from point cloud in LAS 1.3 only
        new_header=laspy.LasHeader(version=version,point_format=formatId)
        scale=0.001
        if formatId in [4,5,9,10]:
            new_header.global_encoding.value=2

        new_header.system_identifier=self.LAS_fmt.identifier["system_identifier"]
        new_header.generating_software=self.LAS_fmt.identifier["generating_software"]
        self.vlrs_list,vlrs_size=pack_VLRbody(self.output_data.metadata['vlrs'])
        new_header.offset_to_point_data=235+vlrs_size

        new_header.mins=np.min(self.output_data.XYZ,axis=0)
        new_header.maxs=np.max(self.output_data.XYZ,axis=0)
        new_header.offsets=np.int64(new_header.mins*scale)/scale
        new_header.scales=np.array([scale]*3)

        new_header.x_scale,new_header.y_scale,new_header.z_scale=new_header.scales
        new_header.x_offset,new_header.y_offset,new_header.z_offset=new_header.offsets
        new_header.x_min,new_header.y_min,new_header.z_min=new_header.mins
        new_header.x_max,new_header.y_max,new_header.z_max=new_header.maxs

        new_header.point_count=len(self.output_data)
                
        pt_return_count=[0]*5
        unique,counts=np.unique(self.output_data.return_number,return_counts=True)
        for i in unique:
            try:
                pt_return_count[i-1]=counts[i-1]
            except: pass
        new_header.number_of_points_by_return=pt_return_count
        return new_header
        
    def writeAttr(self):
        # point_dtype=[('X','int32'),('Y','int32'),('Z','int32')]+self.LAS_fmt.recordFormat[self.point_record.header.point_format.id]
        #writing conventional fields
        coords_int=np.array((self.output_data.XYZ-self.point_record.header.offsets)/self.point_record.header.scales,dtype=np.int32)
        self.point_record.X=coords_int[:,0]
        self.point_record.Y=coords_int[:,1]
        self.point_record.Z=coords_int[:,2]
        for i in self.LAS_fmt.recordFormat[self.point_record.header.point_format.id]:
            try:
                data=getattr(self.output_data,i[0])
                setattr(self.point_record,i[0],getattr(np,i[1])(data))
            except:
                warnings.warn("Warning: Not possible to write attribute : "+i[0])

def readLAS(filepath,extraField=False,parallel=True):
    """Reading LAS with PyLas

    Args:
        filepath (str): input LAS file (extensions: .las or .laz)
        extraField (bool, optional): True if you want to load additional fields. Defaults to False.

    Returns:
        'plateforme_lidar.utils.lasdata': lasdata object
    """
    f=laspy.read(filepath,laz_backend=utils.LASPY_PARALLEL_BACKEND[parallel])
    LAS_fmt=utils.LAS_FORMAT()
    
    metadata={"vlrs":read_VLRbody(f.vlrs),"extraField":[]}
    output=utils.lasdata()

    for i in LAS_fmt.recordFormat[f.header.point_format.id]:
        try:
            output[i[0]]=np.array(getattr(f,i[0]))
        except:
            print("[LasPy] "+str(i[0])+" not found !")      

    output['XYZ']=f.xyz

    if extraField:
        for i in f.point_format.extra_dimension_names:
            name=i.replace('(','').replace(')','').lower()
            metadata['extraField']+=[name]
            output[name]=f[i]

    output['metadata']=metadata       
    return output

# def sortLASdata(data,names,mode='standard'):
#     namesAttr=utils.fields_names[mode]
#     data_sort=np.copy(data)
#     names_sort=np.copy(names)
#     for i in namesAttr:
#         listNames=list(names_sort)
#         idx_true=namesAttr.index(i)+3
#         if i not in names_sort:
#             raise ValueError("Attribute %s isn't present in your column names !" %i)
#         if listNames.index(i)!=idx_true:
#             data_sort[:,[listNames.index(i),idx_true]]=data_sort[:,[idx_true,listNames.index(i)]]
#             names_sort[[listNames.index(i),idx_true]]=names_sort[[idx_true,listNames.index(i)]]
#     return data_sort,names_sort

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
    sizes=np.uint16(lasdata.wavepacket_size)
    offset=np.uint64(lasdata.wavepacket_offset)
    pkt_desc_index=lasdata.wavepacket_index
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

class GPSTime(object):
    def __init__(self,gpstime:list):
        """Manage GPS Time and convert between Adjusted Standard and Week GPS time
        GPS time start on 1980-01-06 00:00:00 UTC
        Args:
            gpstime (list): GPS time
        """
        self.gps_fmt_code=["GPS week time","Adjusted Standard GPS time","Standard GPS time"]
        self.gps_epoch_datetime=datetime(1980,1,6,tzinfo=timezone.utc)
        self.offset_time=int(10**9)
        self.sec_in_week=int(3600*24*7)
        self.gpstime=np.atleast_1d(gpstime)
            
        self.GPSFormat=self.get_format()

    def _get_week_number(self,standardTime):
        """Compute the week number in GPS standard time

        Args:
            standardTime (float or list): timestamp in standard GPS time format

        Raises:
            ValueError: if there are GPS time from different week in list

        Returns:
            int : week number since GPS epoch starting
        """
        if np.ndim(standardTime)==0:
            week_number=int(standardTime//self.sec_in_week)
        else:
            week_num_first=min(standardTime)//self.sec_in_week
            week_num_last=max(standardTime)//self.sec_in_week
            if week_num_first==week_num_last:
                week_number=int(week_num_first)
            else:
                raise ValueError("GPS Time values aren't in same week")
        return week_number
        
    def get_format(self):
        if all(self.gpstime<self.sec_in_week):
            result=self.gps_fmt_code[0]
        elif all(self.gpstime<self.offset_time):
            result=self.gps_fmt_code[1]
        else:
            result=self.gps_fmt_code[2]

        return result

    def adjStd2week(self):
        """Conversion from Adjusted Standard GPS time format to week time

        Raises:
            ValueError: if your data aren't in Adjusted Standard GPS time

        Returns:
            int: week number
            list: list of GPS time in week time format
        """
        if self.GPSFormat!=self.gps_fmt_code[1]:
            raise ValueError("GPS time format is not "+self.gps_fmt_code[1])
        else:
            temp=self.gpstime+self.offset_time
            week_number=self._get_week_number(temp)
            return week_number,temp%self.sec_in_week

    def week2adjStd(self,date_in_week=[],week_number=0):
        """Conversion from week GPS time format to Adjusted Standard time

        Args:
            date_in_week (list, optional): date of project in format (year,month,day). Defaults to [].
            week_number (int, optional): week number. Defaults to 0.

        Raises:
            ValueError: if your data aren't in Week GPS time format
            ValueError: You hav to give at least date_in_week or week_number

        Returns:
            list: list of Adjusted Standard GPS time
        """
        if self.GPSFormat!=self.gps_fmt_code[0]:
            raise ValueError("GPS time format is not "+self.gps_fmt_code[0])
        
        elif len(date_in_week)>0:
            date_datetime=datetime(*date_in_week,tzinfo=timezone.utc)
            week_number=self._get_week_number(date_datetime.timestamp()-self.gps_epoch_datetime.timestamp())

        elif week_number==0:
            raise ValueError("You have to give date_in_week OR week_number")

        adjStd_time=(self.gpstime+week_number*self.sec_in_week)-self.offset_time
        return adjStd_time