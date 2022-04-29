# coding: utf-8
# Baptiste Feldmann
from subprocess import Popen,PIPE,STDOUT
import subprocess,os,struct,datetime,time,re
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

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def snake_to_camel(name):
    return ''.join(word.title() for word in name.split('_'))

class Timing(object):
    def __init__(self,length,step=20):
        self.length=length
        if step>5:
            listVerbose=[2]+list(range(step,100,step))+[98]
        else:
            listVerbose=list(range(step,100,step))+[98]
        self.pourcent=[int(i*self.length/100) for i in listVerbose]
        self.start=time.time()
    def timer(self,idx):
        if idx in self.pourcent:
            duration=round(time.time()-self.start,1)
            remain=round(duration*(self.length/idx-1),1)
            msg=str(idx)+" in "+str(duration)+"sec - remaining "+str(remain)+"sec"
            return msg

#---CloudCompare---#
QUERY_0={"standard":"G:/RENNES1/BaptisteFeldmann/CloudCompare_11022020/CloudCompare -silent",
         "standard_view":"G:/RENNES1/BaptisteFeldmann/CloudCompare_11022020/CloudCompare",
         "PoissonRecon":"G:/RENNES1/BaptisteFeldmann/AdaptiveSolvers/PoissonRecon",
         "CC_PL":"G:/RENNES1/BaptisteFeldmann/CloudCompare_PL_01042022/CloudCompare -silent"}

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
HEADER_WDP_BYTE=struct.pack("=H16sHQ32s",*(0,b'LASF_Spec',65535,0,b'WAVEFORM_DATA_PACKETS'))
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
    def __repr__(self):
        var=len(self.metadata["extraField"])
        return f'<LAS object of {len(self.XYZ)} points with {var} extra-fields>'
    def __getitem__(self,key):
        return self.__dict__[key]
    def __setitem__(self,key,item):
        self.__dict__[key]=item
    pass

class LAS_FORMAT(object):
    def __init__(self):
        std=[("intensity","uint16"),
             ("return_number","uint8"),
             ("number_of_returns","uint8"),
             ("classification","uint8"),
             ("scan_angle_rank","int8"),
             ("user_data","uint8"),
             ("scan_direction_flag","uint8"),
             ("point_source_id","uint16")]

        gps=[("gps_time","float64")]

        rgb=[("red","uint16"),
             ("green","uint16"),
             ("blue","uint16")]
        
        nir=[("nir","uint16")]

        fwf=[("wavepacket_index","uint8"),
             ("wavepacket_offset","uint64"),
             ("wavepacket_size","uint32"),
             ("return_point_wave_location","float32"),
             ("x_t","float32"),
             ("y_t","float32"),
             ("z_t","float32")]

        systemId='ALTM Titan DW 14SEN343'
        softwareId='Lidar Platform by Univ. Rennes 1'

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
CRS_KEY={"Vertical":4096,"Projected":3072}
GEOKEY_STANDARD={1:(1,0,4),1024:(0, 1, 1),3076:(0, 1, 9001), 4099:(0, 1, 9001)}
#=================#

#---PoissonRecon---#
POISSONRECON_PARAMETERS={"bType":{"Free":"1","Dirichlet":"2","Neumann":"3"}}
#==================#

#---PySBF---#
class pointcloud(object):
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
    def __repr__(self):
        var=len(self.metadata["ScalarNames"])
        return f'<SBF object of {len(self.XYZ)} points with {var} attributes>'
    def __getitem__(self,key):
        return self.__dict__[key]
    def __setitem__(self,key,item):
        self.__dict__[key]=item
    pass

convention={"gpstime":"gps_time","numberofreturns":"number_of_returns","returnnumber":"return_number",
            "scananglerank":"scan_angle_rank","pointsourceid":"point_source_id"}
#===================#

#---SBET---#
field_names = ('time (s)', 'latitude (deg)', 'longitude (deg)', 'hauteur (m)', 
                       'x_vel (m/s)', 'y_vel (m/s)', 'z_vel (m/s)', 
                       'roll (rad)', 'pitch (rad)', 'platform_heading (rad)', 'wander_angle (rad)', 
                       'x_acceleration (m/s²)', 'y_acceleration (m/s²)', 'z_acceleration (m/s²)', 
                       'x_angular_rate (rad/s)', 'y_angular_rate (rad/s)', 'z_angular (rad/s)')

LIST_OF_ATTR=[('time','float64'),('latitude','float64'),('longitude','float64'),('elevation','float32'),
        ('x_vel','float32'),('y_vel','float32'),('z_vel','float32'),
        ('roll','float32'),('pitch','float32'),('heading','float32'),('wander_angle','float32'),
        ('x_acceleration','float32'),('y_acceleration','float32'),('z_acceleration','float32'),
        ('x_angular_rate','float32'),('y_angular_rate','float32'),('z_angulare_rate','float32')]
# 17 attributes of 8 bytes each = 136 bytes
LINE_SIZE=int(136)
# vertical datum folder
VERTICAL_DATUM_DIR='G:/RENNES1/BaptisteFeldmann/Vertical_datum/'
#====================#

#---GDAL---#
GDAL_QUERY_ROOT="osgeo4w "
#===============#

#---Other---#
class DATE(object):
    def __init__(self):
        today=datetime.datetime.now().timetuple()
        self.year=today.tm_year
        self.day=today.tm_mday
        self.month=today.tm_mon
        self.date=str(str(self.year)+"-"+str("0"*(2-len(str(self.month)))+str(self.month))+"-"+str("0"*(2-len(str(self.day)))+str(self.day)))
        self.time=str("0"*(2-len(str(today.tm_hour)))+str(today.tm_hour))+"h"+str("0"*(2-len(str(today.tm_min)))+str(today.tm_min))

