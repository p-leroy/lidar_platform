# Paul Leroy

from struct import pack, unpack


def unpack_vlr_extra_bytes(vlr):
    fh = unpack('=2sBB32s4sq16sq16sq16sq16sq16s32s', vlr.record_data)
    {"reserved": fh[0],
     "data_type": fh[1],
     "options": fh[2],
     "name": fh[3],
     "unused": fh[4],
     "no_data": fh[5],
     "deprecated1": fh[6],
     "min": fh[7],
     "deprecated2": fh[8],
     "max": fh[9],
     "deprecated3": fh[10],
     "scale": fh[11],
     "deprecated4": fh[12],
     "offset": fh[13],
     "deprecated5": fh[14],
     "description": fh[15]
     }


def unpack_vlr_record_waveform_packet_descriptor(vlr, asList=False):
    # 26 bytes
    # 1 byte, bits per sample [unsigned char]
    # 1 byte, waveform compression type [unsigned char]
    # 4 bytes, number of samples [unsigned long]
    # 4 bytes, temporal sample spacing [unsigned long]
    # 8 bytes, digitizer gain [double]
    # 8 bytes, digitizer offset [double]
    fh = unpack('=2B2L2d', vlr.record_data)
    if asList:
        return fh[0], fh[1], fh[2], fh[3], fh[4], fh[5]
    else:
        return {"bits_per_sample": fh[0],
                "waveform_compression_type": fh[1],
                "number_of_samples": fh[2],
                "temporal_sample_spacing": fh[3],
                "digitizer_gain": fh[4],
                "digitizer_offset": fh[5]}


def pack_vlr_record_waveform_packet_descriptor(descriptor):
    return pack('=2B2L2d',
                descriptor['bits_per_sample'],
                descriptor['waveform_compression_type'],
                descriptor['number_of_samples'],
                descriptor['temporal_sample_spacing'],
                descriptor['digitizer_gain'],
                descriptor['digitizer_offset'])


def unpack_evlr_record_waveform_data_packet(evlr, asList=False):
    # 60 bytes
    # 2 bytes, Reserved [unsigned short]
    # 16 bytes, User ID char[16]
    # 2 bytes, Record ID [unsigned short]
    # 8 bytes, Record Length After Header [unsigned long long]
    # 32 bytes, Description char[32]
    fh = unpack('=H16sHQ32s', evlr)
    if asList:
        return fh[0], fh[1], fh[2], fh[3], fh[4]
    else:
        return {'reserved': fh[0],
                'user_id': fh[1],
                'record_id': fh[2],
                'record_length_after_header': fh[3],
                'description': fh[4]}


def pack_evlr_record_waveform_data_packet(waveform_data_packet):
    # 60 bytes
    # 2 bytes, Reserved [unsigned short]
    # 16 bytes, User ID char[16]
    # 2 bytes, Record ID [unsigned short]
    # 8 bytes, Record Length After Header [unsigned long long]
    # 32 bytes, Description char[32]
    return pack('=H16sHQ32s',
                waveform_data_packet['reserved'],
                waveform_data_packet['user_id'],
                waveform_data_packet['record_id'],
                waveform_data_packet['record_length_after_header'],
                waveform_data_packet['description'])


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

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, item):
        self.__dict__[key] = item
        pass

class LASFormat(object):
    def __init__(self):
        std = [("intensity", "uint16"),
               ("return_number", "uint8"),
               ("number_of_returns", "uint8"),
               ("classification", "uint8"),
               ("scan_angle_rank", "int8"),  # scan_angle_rank
               ("user_data", "uint8"),
               ("scan_direction_flag", "uint8"),
               ("point_source_id", "uint16")]

        std_6_10 = [("intensity", "uint16"),
               ("return_number", "uint8"),
               ("number_of_returns", "uint8"),
               ("classification", "uint8"),
               ("scan_angle", "int16"),  # scan_angle
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

        pack = [std, std + gps, std + rgb,  # 0 1 2
                std + gps + rgb,  # 3
                std + gps + fwf,  # 4
                std + gps + rgb + fwf,  # 5
                std_6_10 + gps,  # 6
                std_6_10 + gps + rgb,  # 7
                std_6_10 + gps + rgb + nir,  # 8
                std_6_10 + gps + fwf,  # 9
                std_6_10 + gps + rgb + nir + fwf]  # 10

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
