# coding: utf-8
# Baptiste Feldmann

import datetime
import os
import re
import socket
import subprocess
import struct
import time
from subprocess import Popen, PIPE, STDOUT

from numpy import loadtxt


def delete_file(files):
    for file in files:
        try:
            os.remove(file)
        except:
            pass


def run(query, silent=False, opt_shell=False, sleeping=0):
    process = Popen(query, stdout=PIPE, stderr=STDOUT, shell=opt_shell)
    if not silent:
        print(str(process.stdout.readline(), encoding='utf-8'), end='\r')
        while process.poll() is None:
            if len(process.stdout.readline())>0:
                print(str(process.stdout.readline(), encoding='utf-8'), end='\r')
        print(str(process.stdout.read(), encoding='utf-8'))
    else:
        while process.poll() is None:
            continue
    process.wait()
    if sleeping > 0:
        time.sleep(sleeping)


def run_bis(query, shell=False):
    subprocess.run(query, shell=shell)


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


hostname = socket.gethostname()
if hostname == 'LIDAR-SERVER':
    QUERY_0 = {"standard" : 'C:\Program Files\CloudCompare\CloudCompare.exe -silent',
               "standard_view" : 'C:\Program Files\CloudCompare\CloudCompare.exe',
               "PoissonRecon" : "G:/RENNES1/BaptisteFeldmann/AdaptiveSolvers/PoissonRecon.exe",
               "cc_ple" : "G:/RENNES1/BaptisteFeldmann/CloudCompare_PL_01042022/CloudCompare.exe -silent",
               "cc_ple_view" : "G:/RENNES1/BaptisteFeldmann/CloudCompare_PL_01042022/CloudCompare.exe"
               }
else:
    QUERY_0 = {"standard": 'C:\Program Files\CloudCompare\CloudCompare.exe -silent',
               "standard_view": 'C:\Program Files\CloudCompare\CloudCompare.exe',
               "PoissonRecon": 'C:\opt\AdaptiveSolvers.x64\PoissonRecon.exe',
               "cc_ple": 'C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe -silent',
               "cc_ple_view": 'C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe'
               }

EXPORT_FMT = {"LAS" : " -c_export_fmt LAS -ext laz -auto_save OFF",
              "LAS_auto_save" : " -c_export_fmt LAS -ext laz",
              "BIN_auto_save" : " -c_export_fmt BIN",
              "SBF_auto_save" : " -c_export_fmt SBF",
              "PLY_cloud" : " -c_export_fmt PLY -PLY_export_fmt BINARY_LE -auto_save OFF",
              "PLY_mesh" : " -m_export_fmt PLY -PLY_export_fmt BINARY_LE -auto_save OFF",
              "SBF" : " -c_export_fmt SBF -auto_save OFF"}

SHIFT = {}
globalShiftFile = os.path.split(os.path.abspath(__file__))[0] + "\\global_shift.txt"
for i in loadtxt(globalShiftFile, str, delimiter=";"):
    SHIFT[i[0]] = i[1]


# ---LasFWF---#
HEADER_WDP_BYTE=struct.pack("=H16sHQ32s",*(0,b'LASF_Spec',65535,0,b'WAVEFORM_DATA_PACKETS'))


# ---Lastools---#
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
        var = len(self.metadata["extraField"])
        return f'<LAS object of {len(self.XYZ)} points with {var} extra-fields>'

    def __getitem__(self,key):
        return self.__dict__[key]

    def __setitem__(self,key,item):
        self.__dict__[key] = item
        pass


class LASFormat(object):
    def __init__(self):
        std = [("intensity", "uint16"),
               ("return_number", "uint8"),
               ("number_of_returns", "uint8"),
               ("classification", "uint8"),
               ("scan_angle_rank", "int8"),  # scan_angle? scan_angle_rank?
               ("user_data", "uint8"),
               ("scan_direction_flag", "uint8"),
               ("point_source_id", "uint16")]

        gps = [("gps_time", "float64")]

        rgb = [("red", "uint16"),
               ("green", "uint16"),
               ("blue", "uint16")]
        
        nir = [("nir", "uint16")]

        fwf = [("wavepacket_index", "uint8"),
               ("wavepacket_offset", "uint64"),
               ("wavepacket_size", "uint32"),
               ("return_point_wave_location", "float32"),
               ("x_t", "float32"),
               ("y_t", "float32"),
               ("z_t", "float32")]

        system_id = 'ALTM Titan DW 14SEN343'
        software_id = 'Lidar Platform by Univ. Rennes 1'

        pack = [std, std + gps, std + rgb,   # 0 1 2
                std + gps + rgb,   # 3
                std + gps + fwf,   # 4
                std + gps + rgb + fwf,  # 5
                std + gps,  # 6
                std + gps + rgb,  # 7
                std + gps + rgb + nir,  # 8
                std + gps + fwf,  # 9
                std + gps + rgb + nir + fwf]  # 10

        record_len = [20, 28, 26,  # 0 1 2
                      26 + 8,  # 3
                      28 + 29,  # 4
                      26 + 8 + 29,  # 5
                      30,  # 6
                      30 + 6,  # 7
                      30 + 8,  # 8
                      30 + 29,  # 9
                      30 + 6 + 29]  # 10

        self.record_format = dict(zip(range(0, 11), pack))
        self.data_record_len = dict(zip(range(0, 11), record_len))

        format_names = ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'float32', 'float64']
        format_sizes = [1, 1, 2, 2, 4, 4, 8, 8, 4, 8]
        self.fmt_name_value = dict(zip(format_names, range(1, len(format_names) + 1)))
        self.fmt_name_size = dict(zip(format_names, format_sizes))

        self.identifier = {"system_identifier": system_id + '\x00' * (32 - len(system_id)),
                           "generating_software": software_id + '\x00' * (32 - len(software_id))}


# ---VLRS Geokey---#
CRS_KEY = {"Vertical": 4096,
           "Projected": 3072}

GEOKEY_STANDARD = {1: (1, 0, 4),
                   1024: (0, 1, 1),
                   3076: (0, 1, 9001),
                   4099: (0, 1, 9001)}

# ---PoissonRecon---#
POISSON_RECON_PARAMETERS = {"bType": {"Free": "1", "Dirichlet": "2", "Neumann": "3"}}


# ---PySBF---#
class PointCloud(object):
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
        var = len(self.metadata["ScalarNames"])
        return f'<SBF object of {len(self.XYZ)} points with {var} attributes>'

    def __getitem__(self,key):
        return self.__dict__[key]

    def __setitem__(self,key,item):
        self.__dict__[key] = item
    pass


convention = {"gpstime": "gps_time",
              "numberofreturns": "number_of_returns",
              "returnnumber": "return_number",
              "scananglerank": "scan_angle_rank",
              "pointsourceid": "point_source_id"}

# ---SBET---#
field_names = ('time (s)',
               'latitude (deg)',
               'longitude (deg)',
               'hauteur (m)',
               'x_vel (m/s)',
               'y_vel (m/s)',
               'z_vel (m/s)',
               'roll (rad)',
               'pitch (rad)',
               'platform_heading (rad)',
               'wander_angle (rad)',
               'x_acceleration (m/s²)',
               'y_acceleration (m/s²)',
               'z_acceleration (m/s²)',
               'x_angular_rate (rad/s)',
               'y_angular_rate (rad/s)',
               'z_angular (rad/s)')

LIST_OF_ATTR = [('time', 'float64'),
                ('latitude', 'float64'),
                ('longitude', 'float64'),
                ('elevation', 'float32'),
                ('x_vel', 'float32'),
                ('y_vel', 'float32'),
                ('z_vel', 'float32'),
                ('roll', 'float32'),
                ('pitch', 'float32'),
                ('heading', 'float32'),
                ('wander_angle', 'float32'),
                ('x_acceleration', 'float32'),
                ('y_acceleration', 'float32'),
                ('z_acceleration', 'float32'),
                ('x_angular_rate', 'float32'),
                ('y_angular_rate', 'float32'),
                ('z_angular_rate', 'float32')]

# 17 attributes of 8 bytes each = 136 bytes
LINE_SIZE = int(136)

# vertical datum folder
if hostname == 'LIDAR-SERVER':
    VERTICAL_DATUM_DIR = r'G:\RENNES1\BaptisteFeldmann\Vertical_datum'
else:
    VERTICAL_DATUM_DIR = r'C:\DATA\Vertical_datum'

# ---GDAL---#
GDAL_QUERY_ROOT = "osgeo4w "


# ---Other---#
class DATE(object):
    def __init__(self):
        today = datetime.datetime.now().timetuple()
        self.year = today.tm_year
        self.day = today.tm_mday
        self.month = today.tm_mon
        self.date = str(str(self.year) \
                        + "-" + str("0" * (2-len(str(self.month))) \
                                    + str(self.month)) + "-" + str("0" * (2-len(str(self.day)))+str(self.day)))
        self.time = str("0" * (2 - len(str(today.tm_hour)))
                        + str(today.tm_hour)) + "h" + str("0" * (2 - len(str(today.tm_min))) + str(today.tm_min))
