# coding: utf-8
# Baptiste Feldmann
# Paul Leroy

import copy, os, mmap, struct, time
from datetime import datetime, timezone
import logging

import numpy as np
import laspy

from . import las_fmt
from ..tools import misc

logger = logging.getLogger(__name__)
logging.basicConfig()


# LAS full waveform
HEADER_WDP_BYTE = struct.pack("=H16sHQ32s", * (0, b'LASF_Spec', 65535, 0, b'WAVEFORM_DATA_PACKETS'))

# VLRS Geokey
CRS_KEY = {"Vertical": 4096,
           "Projected": 3072}

GEOKEY_STANDARD = {1: (1, 0, 4),
                   1024: (0, 1, 1),
                   3076: (0, 1, 9001),
                   4099: (0, 1, 9001)}

def filter_las(obj, select):
    """Filter lasdata

    Args:
        obj ('plateforme_lidar.utils.lasdata'): lasdata object
        select (list or int): list of boolean, list of integer or integer

    Returns:
        'plateforme_lidar.utils.lasdata': filtering lasdata object
    """

    if type(select) == list or type(select) == np.ndarray:
        if not len(select) == len(obj):
            select = np.array(select)[np.argsort(select)]

    obj_new = las_fmt.lasdata()
    obj_new['metadata'] = obj.metadata
    features = list(obj.__dict__.keys())
    features.remove("metadata")

    for feature in features:
        setattr(obj_new, feature, getattr(obj, feature)[select])

    return obj_new


def merge_las(obj_list):
    """Merge lasdata
    The returned structure takes format of the first in the list
    All the extraFields aren't kept

    Args:
        obj_list (list): list of 'plateforme_lidar.utils.lasdata' type

    Returns:
        'plateforme_lidar.utils.lasdata': lasdata merged
    """
    merge = las_fmt.lasdata()
    merge['metadata'] = copy.deepcopy(obj_list[0].metadata)
    merge['metadata']['extraField'] = []
    feature_list = list(obj_list[0].__dict__.keys())
    [feature_list.remove(i) for i in obj_list[0].metadata["extraField"] + ["metadata"]]
    
    for feature in feature_list:
        merge[feature] = np.concatenate([i[feature] for i in obj_list], axis=0)
    return merge


def filter_wdp(lines, select):
    """Filter WDP tab

    Args:
        lines (list): list of waveforms
        select (list): list of boolean or list of index

    Returns:
        list : list of extracted waveforms
    """
    if len(select) == len(lines):
        select = np.where(select)[0]
    else:
        select = np.array(select)[np.argsort(select)]
    
    return [lines[i] for i in select]


def update_byte_offset(las_obj, waveforms, byte_offset_start=60):
    """Update byte offset to waveform data

    Args:
        las_obj ('plateforme_lidar.utils.lasdata'): LAS dataset to update
        waveforms (list): list of waveforms
        byte_offset_start (int, optional): byte number of first line in WDP file. Defaults to 60.

    Raises:
        ValueError: Las file must have same number of points than waveforms !
    """
    new_offset = [byte_offset_start]
    sizes = np.uint16(las_obj.wavepacket_size)
    if len(las_obj) != len(waveforms):
        raise ValueError("Las file must have same number of points than waveforms !")

    for i in range(0, len(las_obj)):
        new_offset += [np.uint64(new_offset[i]+sizes[i])]
    las_obj.wavepacket_offset = new_offset[0:-1]


def read_vlr_body(vlrs):
    # read VLR in LAS file with PyLas
    # Can only read waveform, bbox tile and projection vlrs
    list_ = {}
    for vlr in vlrs:
        if 100 <= vlr.record_id <= 356:
            #read waveform vlrs
            # (Bits/sample,wavefm compression type,nbr of samples,Temporal spacing,digitizer gain,digitizer offset)
            list_[vlr.record_id] = struct.unpack("=BBLLdd", vlr.record_data_bytes())
        elif vlr.record_id == 10:
            #read bbox tile vlrs :
            # (level,index,implicit_lvl,reversible,buffer,min_x,max_x,min_y,max_y)
            list_[vlr.record_id] = struct.unpack("=2IH2?4f", vlr.record_data)
        elif vlr.record_id == 34735:
            #read Projection
            # (KeyDirectoryVersion,KeyRevision,MinorRevision,NumberofKeys)+ n*(KeyId,TIFFTagLocation,Count,Value_offset)
            geo_key_list = struct.unpack("=4H", vlr.record_data_bytes()[0:8])
            for i in range(0,int((len(vlr.record_data_bytes())-8)/8)):
                temp = struct.unpack("=4H", vlr.record_data_bytes()[8 * (i + 1): 8 * (i + 1) + 8])
                if temp[1] == 0 and temp[2] == 1:
                    geo_key_list += temp
            list_[vlr.record_id] = geo_key_list
    return list_


def pack_vlr_body(dictio):
    # write VLR in LAS file with LasPy
    # Can only write waveform, bbox tile and projection vlrs
    list_ = []
    size = 0
    if len(dictio) > 0:
        for i in dictio.keys():
            if 100 <= i <= 356:
                # waveform vlrs
                # (Bits/sample, waveform compression type, nbr of samples, Temporal spacing,
                # digitizer gain, digitizer offset)
                temp = laspy.header.VLR(user_id="LASF_Spec",
                                        record_id=i,
                                        record_data=struct.pack("=BBLLdd", *dictio[i]))
            elif i == 10:
                # bbox tile vlrs :
                # (level,index,implicit_lvl,reversible,buffer,min_x,max_x,min_y,max_y)
                temp=laspy.header.VLR(user_id="LAStools",
                                      record_id=i,
                                      record_data=struct.pack("=2IH2?4f", *dictio[i]))
            elif i == 34735:
                # Projection
                # (KeyDirectoryVersion, KeyRevision, MinorRevision, NumberofKeys)
                # + n * (KeyId,TIFFTagLocation,Count,Value_offset)
                fmt = "="+str(len(dictio[i]))+"H"
                temp = laspy.header.VLR(user_id="LASF_Projection",
                                        record_id=i,
                                        record_data=struct.pack(fmt, *dictio[i]))
            else:
                raise Exception("VLR.record_id unknown : "+str(i))

            size += len(temp.record_data_bytes())
            list_ += [temp]
    return list_, size


def vlrs_keys(vlrs, geokey):
    """Add geokey in VLR Projection

    Args:
        vlrs (dict): lasdata.metadata['vlrs']
        geokey (dict): geokey={"Vertical":epsg,"Projected":epsg}

    Returns:
        dict: updated vlrs
    """
    vlrs_copy = vlrs.copy()
    if 34735 in vlrs.keys():
        vlrs_dict = {}
        for i in range(0, int(len(vlrs[34735]) / 4)):
            num = i * 4
            vlrs_dict[vlrs[34735][num]] = vlrs[34735][num + 1: num + 4]
    else:
        vlrs_dict = GEOKEY_STANDARD

    for i in list(geokey.keys()):
        vlrs_dict[CRS_KEY[i]] = [0, 1, geokey[i]]

    vlrs_sort = np.sort(list(vlrs_dict.keys()))
    vlrs_final = []
    for i in vlrs_sort:
        vlrs_final += [i]
        vlrs_final += vlrs_dict[i]
    vlrs_final[3] = len(vlrs_sort) - 1
    vlrs_copy[34735] = tuple(vlrs_final)
    return vlrs_copy


class WriteLAS(object):
    def __init__(self, filepath, data, format_id=1, extra_fields=[], waveforms=[], parallel=True):
        """Write LAS 1.3 with laspy

        Args:
            filepath (str): output file path (extensions= .las or .laz)
            data ('plateforme_lidar.utils.lasdata'): lasdata object
            format_id (int, optional): data format id according to ASPRS convention (standard mode=1, fwf mode=4). Defaults to 1.
            extraField (list, optional): list of additional fields [(("name","type format"),listData),...]
                ex: [(("depth","float32"),numpy.ndarray),(("value","uint8"),numpy.ndarray)]. Defaults to [].
            waveforms (list, optional): list of waveforms to save in external WDP file. Make sure that format_id is compatible with wave packet (ie. 4,5,9 or 10). Default to []
        """
        # standard : format_id=1 ; fwf : format_id=4
        print("[Writing LAS file]..", end="")
        self._start = time.time()
        self.output_data = data
        self.LAS_fmt = las_fmt.LASFormat()
        # new_header=self.createHeader("1.3",format_id)
        # pointFormat=laspy.PointFormat(format_id)
        # for extraField in extraFields:
        #     pointFormat.add_extra_dimension(laspy.ExtraBytesParams(name=extraField["name"],type=extraField["type"],description="Extras_fields"))
        # new_points=laspy.PackedPointRecord(points,point_format=pointFormat)

        self.point_record = laspy.LasData(header=self.create_header("1.3", format_id),
                                          points=laspy.ScaleAwarePointRecord.zeros(
                                              len(self.output_data),
                                              header=self.create_header("1.3", format_id))
                                          )

        for extraField in extra_fields:
            name_ = extraField[0][0]
            type_ = extraField[0][1]
            data_ = extraField[1]
            self.point_record.add_extra_dim(laspy.ExtraBytesParams(name=name_,
                                                                   type=getattr(np, type_),
                                                                   description="Extra_fields"))
            setattr(self.point_record, name_, data_)

        self.write_attr()
        if format_id in [4, 5, 9, 10]:
            backend = laspy.compression.LazBackend(2)
        else:
            backend = laspy.compression.LazBackend(int(not parallel))
        self.point_record.write(filepath, laz_backend=backend)
        print("done")

        if len(waveforms) > 0 and format_id in [4, 5, 9, 10]:
            self.wave_data_packet(filepath, waveforms)
    
    def __repr__(self):
        return "Write "+str(len(self.output_data))+" points in "+str(round(time.time()-self._start,1))+" sec"

    def wave_data_packet(self, filepath, waveforms):
        # write external waveforms in WDP file not compressed
        # Future improvement will make writing compressed possible
        nbrPoints = len(self.output_data)
        sizes = np.uint16(self.output_data.wavepacket_size)
        offsets = np.uint64(self.output_data.wavepacket_offset)
        pkt_desc_index = self.output_data.wavepacket_index
        vlrs = self.output_data.metadata['vlrs']

        if not all(offsets[1::]==(offsets[0:-1]+sizes[0:-1])):
            raise ValueError("byte offset list is not continuous, re-compute your LAS dataset")
        
        start = time.time()
        print("[Writing waveform data packet] %d waveforms" %len(waveforms))
        displayer = misc.Timing(nbrPoints, 20)
        with open(filepath[0:-4]+".wdp","wb") as wdpFile :
            wdpFile.write(HEADER_WDP_BYTE)
            for i in range(0,nbrPoints):
                msg=displayer.timer(i)
                if msg is not None:
                    print("[Writing waveform data packet] "+msg)
                
                if len(waveforms[i]) != (sizes[i] / 2):
                    raise ValueError("Size of waveform n°"+str(i)+" is not the same in LAS file")
                
                try:
                    vlr_body = vlrs[pkt_desc_index[i] + 99]
                except:
                    raise ValueError("Number of the wave packet desc index not in VLRS")

                length = int(vlr_body[2])
        
                try:
                    test = struct.pack(str(length)+'h',*np.int16(waveforms[i]))
                    wdpFile.write(test)
                except:
                    raise ValueError(str(length))
        print("[Writing waveform data packet] done in %d sec" %(time.time()-start))

    def create_header(self, version, formatId):
        #Create header from point cloud in LAS 1.3 only
        new_header = laspy.LasHeader(version=version, point_format=formatId)
        scale = 0.001
        if formatId in [4, 5, 9, 10]:
            new_header.global_encoding.value = 2

        new_header.system_identifier = self.LAS_fmt.identifier["system_identifier"]
        new_header.generating_software = self.LAS_fmt.identifier["generating_software"]
        new_header.vlrs, vlrs_size = pack_vlr_body(self.output_data.metadata['vlrs'])
        new_header.offset_to_point_data = 235 + vlrs_size

        new_header.mins = np.min(self.output_data.XYZ, axis=0)
        new_header.maxs = np.max(self.output_data.XYZ, axis=0)
        new_header.offsets = np.int64(new_header.mins*scale)/scale
        new_header.scales = np.array([scale]*3)

        new_header.x_scale, new_header.y_scale, new_header.z_scale = new_header.scales
        new_header.x_offset, new_header.y_offset, new_header.z_offset = new_header.offsets
        new_header.x_min, new_header.y_min, new_header.z_min = new_header.mins
        new_header.x_max, new_header.y_max, new_header.z_max = new_header.maxs

        new_header.point_count = len(self.output_data)
                
        pt_return_count = [0] * 5
        unique,counts = np.unique(self.output_data.return_number, return_counts=True)
        for i in unique:
            try:
                pt_return_count[i-1] = counts[i-1]
            except: pass
        new_header.number_of_points_by_return = pt_return_count
        return new_header
        
    def write_attr(self):
        # point_dtype=[('X','int32'),('Y','int32'),('Z','int32')]
        # +self.LAS_fmt.recordFormat[self.point_record.header.point_format.id]
        # write conventional fields
        coords_int = np.array((self.output_data.XYZ - self.point_record.header.offsets) / self.point_record.header.scales, dtype=np.int32)
        self.point_record.X = coords_int[:, 0]
        self.point_record.Y = coords_int[:, 1]
        self.point_record.Z = coords_int[:, 2]
        for i in self.LAS_fmt.record_format[self.point_record.header.point_format.id]:
            try:
                data = getattr(self.output_data, i[0])
                setattr(self.point_record, i[0], getattr(np, i[1])(data))
            except:
                print(f'[lastools.WriteLAS.writeAttr] {i[0]} {i[1]}')
                print("[lastools.WriteLAS.writeAttr] Warning: not possible to write attribute: " + i[0])


def read(filepath, extra_fields=False, parallel=True):
    """Read a LAS file with laspy

    Args:
        filepath (str): input LAS file (extensions: .las or .laz)
        extra_fields (bool, optional): True if you want to load additional fields. Defaults to False.
        parallel
        backend (str, optional): 'multi' for parallel backend, 'single' for single-thread mode, 'laszip' to use laszip for LAS fwf

    Returns:
        'plateforme_lidar.utils.lasdata': lasdata object
    """

    point_format = laspy.open(filepath, mode='r', laz_backend=laspy.compression.LazBackend(2)).header.point_format.id

    if point_format in [4, 5, 9, 10]:  # Wave packets
        # Point Data Record Format 4 adds Wave Packets to Point Data Record Format 1
        # Point Data Record Format 5 adds Wave Packets to Point Data Record Format 3
        # Point Data Record Format 9 adds Wave Packets to Point Data Record Format 6
        # Point Data Record Format 10 adds Wave Packets to Point Data Record Format 7
        backend = laspy.compression.LazBackend(2)
    else:
        backend = laspy.compression.LazBackend(int(not parallel))

    las = laspy.read(filepath, laz_backend=backend)
    gps_time_type = las.header.global_encoding.gps_time_type
    print(f'[lastools.ReadLAS] gps_time_type read in header: {gps_time_type.name}')
    
    metadata = {"vlrs": read_vlr_body(las.vlrs),
                "extraField": [],
                'filepath': filepath}
    output = las_fmt.lasdata()
    LAS_fmt = las_fmt.LASFormat()

    for field, dtype in LAS_fmt.record_format[point_format]:
        try:
            output[field] = np.array(getattr(las, field), dtype=dtype)
        except:
            print("[LasPy] " + str(field) + " not found")

    output['XYZ'] = las.xyz

    if extra_fields:
        for extra_dimension_name in las.point_format.extra_dimension_names:
            name = extra_dimension_name.replace('(', '').replace(')', '').replace(' ', '_').lower()
            print(f'rename extra field {extra_dimension_name} to {name}')
            metadata['extraField'] += [name]
            output[name] = las[extra_dimension_name]

    if 'GpsTime' in las.point_format.extra_dimension_names:
        print('[lastools.ReadLAS] WARNING GpsTime found in extra_dimension_names (CloudCompare convention)')
        print('[lastools.ReadLAS] replace standard field gps_time by extra field GpsTime')
        output['gps_time'] = las['GpsTime']

    output['metadata'] = metadata

    return output


def read_wdp(las_data):
    """Reading waveforms in WDP file

    Args:
        las_data ('plateforme_lidar.utils.lasdata'): lasdata object

    Raises:
        ValueError: if for one point wave data packet descriptor is not in VLRS

    Returns:
        list: list of waveform (length of each waveform can be different)
    """
    point_number = len(las_data)
    sizes = np.uint16(las_data.wavepacket_size)
    offset = np.uint64(las_data.wavepacket_offset)
    pkt_desc_index = las_data.wavepacket_index
    vlrs = las_data.metadata['vlrs']
    start = time.time()
    print("[Reading waveform data packet] %d waveforms" %point_number)
    
    with open(las_data.metadata['filepath'][0:-4] + ".wdp", 'rb') as wdp:
        dataraw = mmap.mmap(wdp.fileno(),
                            os.path.getsize(las_data.metadata['filepath'][0:-4] + ".wdp"),
                            access=mmap.ACCESS_READ)

    lines = []
    displayer = misc.Timing(point_number, 20)
    for i in range(0,point_number):
        msg = displayer.timer(i)
        if msg is not None:
            print("[Reading waveform data packet] " + msg)
        
        try:
            vlr_body = vlrs[pkt_desc_index[i] + 99]
        except:
            raise ValueError("Number of the wave packet desc index not in VLRS !")
        
        length = int(vlr_body[2])
        line = np.array(struct.unpack(str(length) + 'h', dataraw[offset[i]: (offset[i] + sizes[i])]))
        lines += [np.round_(line * vlr_body[4] + vlr_body[5], decimals=2)]
    print("[Reading waveform data packet] done in %d sec" %(time.time()-start))
    return lines


def read_ortho_fwf(workspace, las_file):
    print("[Read waveform data packet] : ",end='\r')
    f = laspy.file.File(workspace + las_file)
    nbr_pts = int(f.header.count)
    percent = [int(0.2 * nbr_pts), int(0.4 * nbr_pts), int(0.6 * nbr_pts), int(0.8 * nbr_pts), int(0.95 * nbr_pts)]
    try :
        sizes = np.int_(f.waveform_packet_size)
    except :
        sizes = np.int_(f.points['point']['wavefm_pkt_size'])
    
    offset = np.int_(f.byte_offset_to_waveform_data)
    
    wdp = open(workspace + las_file[0:-4] + ".wdp", 'rb')
    data = mmap.mmap(wdp.fileno(),
                     0,
                     access=mmap.ACCESS_READ)
    temp = read_vlr_body(f.header.vlrs)
    if len(f.header.vlrs) == 1:
        vlr_body = temp[list(temp.keys())[0]]
    else:
        vlr_body = temp[f.header.vlrs[f.wave_packet_desc_index[0]].record_id]
        
    anchor_z = f.z_t*f.return_point_waveform_loc
    step_z = f.z_t[0]*vlr_body[3]
    lines = []
    length = int(vlr_body[2])
    prof = [np.round_(anchor_z-(step_z * c), decimals=2) for c in range(0, length)]
    prof = np.transpose(np.reshape(prof, np.shape(prof)))
    for i in range(0, nbr_pts):
        if i in percent:
            print("%d%%-" %(25+25*percent.index(i)), end='\r')

        line = np.array(struct.unpack(str(length) + 'h',data[offset[i]:offset[i] + sizes[i]]))
        lines += [np.round_(line*vlr_body[4] + vlr_body[5], decimals=2)]
    
    wdp.close()
    f.close()
    print("done !")
    return np.stack([lines,prof]), vlr_body[3], np.round_(step_z, decimals=2)


class GPSTime(object):
    def __init__(self, gpstime: list):
        """Manage GPS Time and convert between Adjusted Standard and Week GPS time
        GPS time start on 1980-01-06 00:00:00 UTC
        Time stamps are either stored in GPS week seconds or Adjusted Standard GPS Time
        (i.e., Standard GPS Time - 1 * 10**9 sec) depending on the "Global Encoding Bit 0" of the LAS Public Header
        Args:
            gpstime (list): GPS time
        """

        self.gps_epoch_datetime = datetime(1980, 1, 6, tzinfo=timezone.utc)
        self.offset_time = int(10 ** 9)
        self.sec_in_week = int(3600 * 24 * 7)
        self.gpstime = np.atleast_1d(gpstime)
        self.gps_time_type = self.get_gps_time_type()

    def __repr__(self):
        return self.gps_time_type

    def _get_week_number(self, standard_time):
        """Compute the week number in GPS standard time

        Args:
            standard_time (float or list): timestamp in standard GPS time format

        Raises:
            ValueError: if there are GPS time from different week in list

        Returns:
            int : week number since GPS epoch starting
        """
        if np.ndim(standard_time) == 0:
            week_number = int(standard_time // self.sec_in_week)
        else:
            week_num_first = min(standard_time) // self.sec_in_week
            week_num_last = max(standard_time) // self.sec_in_week
            if week_num_first == week_num_last:
                week_number = int(week_num_first)
            else:
                print(f'week_num_first {week_num_first}, week_num_last {week_num_last}')
                raise ValueError("[lastools.GPSTime._get_week_number] Time values aren't in same week")
        return week_number
        
    def get_gps_time_type(self):
        if all(self.gpstime < self.sec_in_week):
            gps_time_type = laspy.header.GpsTimeType.WEEK_TIME
        elif all(self.gpstime < self.offset_time):
            gps_time_type = laspy.header.GpsTimeType.STANDARD
        else:
            raise ValueError("[lastools.GPSTime.get_format] Unexpected gps_time_type, neither WEEK_TIME nor STANDARD")

        return gps_time_type

    def adjusted_standard_2_week_time(self):
        """Conversion from Adjusted Standard GPS time format to week time

        Raises:
            ValueError: if your data aren't in Adjusted Standard GPS time

        Returns:
            int: week number
            list: list of GPS time in week time format
        """
        if self.gps_time_type != laspy.header.GpsTimeType.STANDARD:
            raise ValueError("GPS time format is not " + laspy.header.GpsTimeType.STANDARD.name)
        else:
            temp = self.gpstime + self.offset_time
            week_number = self._get_week_number(temp)
            return week_number, temp % self.sec_in_week

    def week_time_2_adjusted_standard(self, date_in_week=[], week_number=0):
        """Conversion from week GPS time format to Adjusted Standard time

        Args:
            date_in_week (list, optional): date of project in format (year, month, day). Defaults to [].
            week_number (int, optional): week number. Defaults to 0.

        Raises:
            ValueError: if your data aren't in Week GPS time format
            ValueError: You have to give at least date_in_week or week_number

        Returns:
            list: list of Adjusted Standard GPS time
        """
        if self.gps_time_type != laspy.header.GpsTimeType.WEEK_TIME:
            raise ValueError("GPS time format is not " + laspy.header.GpsTimeType.WEEK_TIME.name)
        
        elif len(date_in_week) > 0:
            date_datetime = datetime(*date_in_week, tzinfo=timezone.utc)
            week_number = self._get_week_number(date_datetime.timestamp() - self.gps_epoch_datetime.timestamp())

        elif week_number == 0:
            raise ValueError("You have to give date_in_week OR week_number")

        adjusted_standard_time = (self.gpstime + week_number * self.sec_in_week) - self.offset_time
        return adjusted_standard_time
