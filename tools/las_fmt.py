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

        pack = [std, std + gps, std + rgb,  # 0 1 2
                std + gps + rgb,  # 3
                std + gps + fwf,  # 4
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
