# coding: utf-8
# Baptiste Feldmann
from subprocess import Popen,PIPE,STDOUT
import subprocess,os,struct,datetime,time
from numpy import loadtxt

def delete_file(liste):
    for i in liste:
        try:
            os.remove(i)
        except:
            pass

def Run(query,silent=False,optShell=False,sleeping=0):
    process=Popen(query,stdout=PIPE,stderr=STDOUT,shell=optShell)
    if not silent:
        print(str(process.stdout.readline(),encoding='utf-8'),end='\r')
        while process.poll() is None:
            if len(process.stdout.readline())>0:
                print(str(process.stdout.readline(),encoding='utf-8'),end='\r')
        print(str(process.stdout.read(),encoding='utf-8'))
    else:
        while process.poll() is None:
            continue
    process.wait()
    if sleeping>0:
        time.sleep(sleeping)

def Run_bis(query,optShell=False):
    subprocess.run(query,shell=optShell)

class Timing(object):
    def __init__(self,length,step=20):
        self.length=length
        listVerbose=list(range(step,100,step))+[98]
        self.pourcent=[int(i*self.length/100) for i in listVerbose]
        self.start=time.time()
    def timer(self,idx):
        if idx in self.pourcent:
            pos=self.pourcent.index(idx)
            duration=round(time.time()-self.start,1)
            remain=round(duration*(len(self.pourcent)-pos)/(pos+1),1)
            msg=str(idx)+" in "+str(duration)+"sec - remaining "+str(remain)+"sec"
        else:
            msg=""
        return msg

#---CloudCompare---#
QUERY_0={"standard":"G:/RENNES1/BaptisteFeldmann/CloudCompare_11022020/CloudCompare -silent",
         "standard_view":"G:/RENNES1/BaptisteFeldmann/CloudCompare_11022020/CloudCompare",
         "PoissonRecon":"G:/RENNES1/BaptisteFeldmann/AdaptiveSolvers/PoissonRecon"}

EXPORT_FMT={"LAS":" -c_export_fmt LAS -ext laz -auto_save OFF",
            "PLY_cloud":" -c_export_fmt PLY -PLY_export_fmt BINARY_LE -auto_save OFF",
            "PLY_mesh":" -m_export_fmt PLY -PLY_export_fmt BINARY_LE -auto_save OFF",
            "SBF":" -c_export_fmt SBF -auto_save OFF"}

SHIFT={}
globalShiftFile=os.path.split(os.path.abspath(__file__))[0]+"\\global_shift.txt"
for i in loadtxt(globalShiftFile,str,delimiter=";"):
    SHIFT[i[0]]=i[1]
#================#

#---LasFWF---#
headerWDP_binary=struct.pack("=H16sHQ32s",*(0,b'LASF_Spec',65535,0,b'WAVEFORM_DATA_PACKETS'))
#================#

#---Lastools---#
class lasdata(object):
    """LAS data object

    Attributes:
        metadata (dict): {'vlrs': dict (info about LAS vlrs),'extraField': list (list of additional fields)}
        XYZ (numpy.ndarray): coordinates
        various attr (numpy.ndarray):
    
    Functionality:
        len('plateforme_lidar.utils.lasdata'): number of points
        print('plateforme_lidar.utils.lasdata'): list of attributes
        get attribute: lasdata.attribute or lasdata[attribute]
        set attribute: lasdata.attribute=value or lasdata[attribute]=value
        create attribute: setattr(lasdata,attribute,value) or lasdata[attribute]=value
    """   
    def __len__(self):
        return len(self.XYZ)
    def __str__(self):
        return "\n".join(self.__dict__.keys())
    def __getitem__(self,key):
        return self.__dict__[key]
    def __setitem__(self,key,item):
        self.__dict__[key]=item
    pass

class LAS_format(object):
    def __init__(self):
        std=[("intensity","uint16"),
             ("return_num","uint8"),
             ("num_returns","uint8"),
             ("classification","uint8"),
             ("scan_angle_rank","int8"),
             ("user_data","uint8"),
             ("scan_dir_flag","uint8"),
             ("pt_src_id","uint16")]

        gps=[("gps_time","float64")]

        rgb=[("red","uint16"),
             ("green","uint16"),
             ("blue","uint16")]
        
        nir=[("nir","uint16")]

        fwf=[("wave_packet_desc_index","uint8"),
             ("byte_offset_to_waveform_data","uint64"),
             ("waveform_packet_size","uint32"),
             ("return_point_waveform_loc","float32"),
             ("x_t","float32"),
             ("y_t","float32"),
             ("z_t","float32")]

        systemId='ALTM Titan DW 14SEN343'
        softwareId='Nantes-Rennes Lidar Platform'

        pack=[std,std+gps,std+rgb,std+gps+rgb,std+gps+fwf,std+gps+rgb+fwf,
              std+gps,std+gps+rgb,std+gps+rgb+nir,std+gps+fwf,std+gps+rgb+nir+fwf]
        recordLen=[20,28,26,26+8,28+29,26+8+29,
                   30,30+6,30+8,30+29,30+6+29]

        self.recordFormat=dict(zip(range(0,11),pack))
        self.dataRecordLen=dict(zip(range(0,11),recordLen))

        format_names=['uint8','int8','uint16','int16','uint32','int32','uint64','int64','float32','float64']
        format_sizes=[1,1,2,2,4,4,8,8,4,8]
        self.fmtNameValue=dict(zip(format_names,range(1,len(format_names)+1)))
        self.fmtNameSize=dict(zip(format_names,format_sizes))

        self.identifier={"system_identifier":systemId+'\x00'*(32-len(systemId)),"generating_software":softwareId+'\x00'*(32-len(softwareId))}

#================#

#---VLRS Geokey---#
CRS_key={"Vertical":4096,"Projected":3072}
geokey_standard={1:(1,0,4),1024:(0, 1, 1),3076:(0, 1, 9001), 4099:(0, 1, 9001)}
#=================#

#---PoissonRecon---#
PoissonRecon_parameters={"bType":{"Free":"1","Dirichlet":"2","Neumann":"3"}}
#==================#

#---PySBF---#
convention={"gpstime":"gps_time","numberofreturns":"num_returns","returnnumber":"return_num",
            "scananglerank":"scan_angle_rank","pointsourceid":"pt_src_id"}
#===================#

#---SBET---#
field_names = ('time (s)', 'latitude (deg)', 'longitude (deg)', 'hauteur (m)', 
                       'x_vel (m/s)', 'y_vel (m/s)', 'z_vel (m/s)', 
                       'roll (rad)', 'pitch (rad)', 'platform_heading (rad)', 'wander_angle (rad)', 
                       'x_acceleration (m/s²)', 'y_acceleration (m/s²)', 'z_acceleration (m/s²)', 
                       'x_angular_rate (rad/s)', 'y_angular_rate (rad/s)', 'z_angular (rad/s)')
#====================#

#---GDAL---#
gdalQueryRoot="osgeo4w "
#===============#

#---Other---#
class date(object):
    def __init__(self):
        today=datetime.datetime.now().timetuple()
        self.year=today.tm_year
        self.day=today.tm_mday
        self.month=today.tm_mon
        self.date=str(str(self.year)+"-"+str("0"*(2-len(str(self.month)))+str(self.month))+"-"+str("0"*(2-len(str(self.day)))+str(self.day)))
        self.time=str("0"*(2-len(str(today.tm_hour)))+str(today.tm_hour))+"h"+str("0"*(2-len(str(today.tm_min)))+str(today.tm_min))

