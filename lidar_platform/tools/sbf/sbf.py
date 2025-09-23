import configparser
import logging
import os
import struct

import numpy as np

logger = logging.getLogger(__name__)


def shift_array(array, shift, config=None, debug=False):
    newArray = array.astype(float)
    # apply the shift read in the SBF file
    newArray += np.array(shift).reshape(1, -1)
    # apply GlobalShift if any
    if config is not None:
        try:
            globalShift = eval(config['SBF']['GlobalShift'])
            logger.debug(f'use GlobalShift {globalShift}')
            newArray += np.array(globalShift).reshape(1, -1)
        except:
            pass
    return newArray


def read_sbf_header(sbf, verbose=False):
    config = configparser.ConfigParser()
    config.optionxform = str
    with open(sbf) as f:
        config.read_file(f)
        if 'SBF' not in config:
            print('sbf badly formatted, no [SBF] section')
        else:
            return config


def write_sbf(sbf, xyz,
              sf=None, config=None, add_index=False, normals=None, global_shift=None):

    path_to_sbf = sbf
    path_to_sbf_data = sbf + '.data'
    n_points = xyz.shape[0]
    if sf is not None:
        sf = sf.reshape(n_points, -1)
        SFCount = sf.shape[1]
    else:
        SFCount = 0

    if global_shift is not None:
        global_shift_str = f'{global_shift[0]}, {global_shift[1]}, {global_shift[2]}'

    # write .sbf
    Points = xyz.shape[0]
    if config is None:  # if there is no config, build one
        dict_SF = {f'SF{k + 1}': f'{k + 1}' for k in range(SFCount)}
        config = configparser.ConfigParser()
        config.optionxform = str
        config['SBF'] = {'Points': str(Points),
                         'SFCount': str(SFCount),
                         **dict_SF}
        if global_shift is not None:
            config['SBF']['GlobalShift'] = global_shift_str
    else:
        config['SBF']['Points'] = str(Points)  # enforce the coherence of the number of points
        config['SBF']['SFCount'] = str(SFCount)
        if global_shift is not None:
            if 'GlobalShift' in config['SBF']:
                config_global_shift = config['SBF']['GlobalShift']
                print(f'[write_sbf] warning: global_shift parameter {global_shift}')
                print(f'            will overwrite the config one {config_global_shift}')
                config['SBF']['GlobalShift'] = global_shift_str

    if add_index is True:
        if 'SFCount' in config['SBF']:
            SFCount += 1
        else:
            SFCount = 1
        config['SBF']['SFcount'] = str(SFCount)
        config['SBF'][f'SF{SFCount}'] = 'index'

    if normals is not None:
        if 'SFCount' in config['SBF']:
            SFCount += 3
        else:
            SFCount = 3
        config['SBF']['SFcount'] = str(SFCount)
        config['SBF'][f'SF{SFCount + 1}'] = 'Nx'
        config['SBF'][f'SF{SFCount + 2}'] = 'Ny'
        config['SBF'][f'SF{SFCount + 3}'] = 'Nz'

    # write .sbf configuration file
    with open(path_to_sbf, 'w') as sbf:
        config.write(sbf)

    # remove GlobalShift
    if 'GlobalShift' in config['SBF']:
        globalShift = eval(config['SBF']['GlobalShift'])
        xyz_orig = xyz - np.array(globalShift).reshape(1, -1)
    else:
        xyz_orig = xyz
    # compute sbf internal shift
    shift = np.mean(xyz_orig, axis=0).astype(float)
    # build the array that will effectively be stored (32 bits float)
    a = np.zeros((Points, SFCount + 3)).astype('>f')
    # set xyz
    a[:, :3] = (xyz_orig - shift).astype('>f')

    # subtract shifts if any configured
    if SFCount != 0:
        shifted_sf = sf.copy()
        for k in range(SFCount):
            i_sf = k + 1
            for item in config['SBF'][f'SF{i_sf}'].split(','):
                if 's=' in item:  # apply offset if any
                    sf_shift = float(item.replace('"', '').split('s=')[1])
                    if verbose:
                        print(f'[read_sbf] subtract shift from scalar field SF{i_sf} for storage: {sf_shift}')
                    shifted_sf[:, k] -= sf_shift
        # set scalar fields
        a[:, 3:] = shifted_sf.astype('>f')

    if add_index is True:
        b = np.zeros((Points, SFCount + 1)).astype('>f')
        b[:, :-1] = a
        b[:, -1] = np.arange(Points).astype('>f')
        a = b

    # write .sbf.data
    with open(path_to_sbf_data, 'wb') as sbf_data:
        # 0-1 SBF header flag
        flag = bytearray([42, 42])
        sbf_data.write(flag)
        # 2-9 Point count (Np)
        sbf_data.write(struct.pack('>Q', Points))
        # 10-11 ScalarField count (Ns)
        sbf_data.write(struct.pack('>H', SFCount))
        # 12-19 X coordinate shift
        sbf_data.write(struct.pack('>d', shift[0]))
        # 20-27 Y coordinate shift
        sbf_data.write(struct.pack('>d', shift[1]))
        # 28-35 Z coordinate shift
        sbf_data.write(struct.pack('>d', shift[2]))
        # 36-63 Reserved for later
        sbf_data.write(bytes(63 - 36 + 1))
        sbf_data.write(a)


def is_int(str_):
    try:
        int(str_)
        return True
    except ValueError:
        return False


class SbfData:

    def __init__(self, filename, verbose=True):

        self.Np = None
        self.filename = filename
        self.xyz = None
        self.sf = None
        self.config = None

        self.read_sbf(verbose=verbose)  # set xyz, sf and config

        self.sf_names = self.get_sf_names()

        self.name_index = self.get_name_index_dict()
        for name, index in self.name_index.items():
            name = name.replace(" ", "_").replace("(", "I").replace(")", "I").replace("@", "_at_").replace('.', 'o')
            self.__setattr__(name, self.sf[:, index])

        self.__setattr__('x', self.xyz[:, 0])
        self.__setattr__('y', self.xyz[:, 1])
        self.__setattr__('z', self.xyz[:, 2])

    def read_sbf(self, verbose=False):
        config = read_sbf_header(self.filename, verbose=verbose)  # READ .sbf header

        ################
        # READ .sbf.data
        # be careful, sys.byteorder is probably 'little' (different from Cloud Compare)
        sbf_data = self.filename + '.data'
        with open(sbf_data, 'rb') as f:
            bytes_ = f.read(64)
            # 0-1 SBF header flag
            flag = bytes_[0:2]
            # 2-9 Point count (Np)
            Np = struct.unpack('>Q', bytes_[2:10])[0]
            # 10-11 ScalarField count (Ns)
            Ns = struct.unpack('>H', bytes_[10:12])[0]
            if verbose is True:
                print(f'flag {flag}, Np {Np}, Ns {Ns}')
            # 12-19 X coordinate shift
            x_shift = struct.unpack('>d', bytes_[12:20])[0]
            # 20-27 Y coordinate shift
            y_shift = struct.unpack('>d', bytes_[20:28])[0]
            # 28-35 Z coordinate shift
            z_shift = struct.unpack('>d', bytes_[28:36])[0]
            # 36-63 Reserved for later
            if verbose is True:
                print(f'shift ({x_shift, y_shift, z_shift})')
                print(bytes_[37:])
                print(len(bytes_[37:]))
            array = np.fromfile(f, dtype='>f').reshape(Np, Ns + 3).astype(float)
            shift = np.array((x_shift, y_shift, z_shift)).reshape(1, 3)

        self.Np = Np

        # shift point cloud
        xyz = shift_array(array[:, :3], shift, config)

        # get scalar fields if any and handle shifts, aka "s=value", if any
        if Ns != 0:
            for k in range(1, Ns + 1):
                for item in config['SBF'][f'SF{k}'].split(','):
                    if 's=' in item:  # apply offset if any
                        shift = float(item.replace('"', '').split('s=')[1])
                        print(f'[read_sbf] add shift to scalar field SF{k}: {shift}')
                        array[:, 2 + k] += shift
            sf = array[:, 3:]
        else:
            sf = None

        self.xyz = xyz

        self.sf = sf

        self.config = config

    def get_name_index_dict(self):
        dict_ = {self.config['SBF'][name].split(',')[0]: int(name.split('SF')[1]) - 1  # remove s= and p= information
                 for name in self.config['SBF'] if len(name.split('SF')) == 2 and is_int(name.split('SF')[1])}
        return dict_

    def get_sf_names(self):
        list_ = [self.config['SBF'][name]
                 for name in self.config['SBF'] if len(name.split('SF')) == 2 and is_int(name.split('SF')[1])]
        list_ = [name.split(',')[0] for name in list_]  # remove s= and p= information
        return list_

    def remove_sf(self, name):
        # remove the scalar field from the sf array
        index = self.name_index[name]
        new_sf = np.delete(self.sf, index, axis=1)
        # copy the configuration
        new_config = configparser.ConfigParser()
        new_config.optionxform = str
        new_config.read_dict(self.config)
        sf_index = index + 1
        sf_count = int(self.config['SBF']['SFCount'])
        # update the configuration
        new_config['SBF']['SFCount'] = str(sf_count - 1)  # decrease the counter of scalar fields
        new_config.remove_option('SBF', f'SF{sf_count}')  # remove the last option
        for idx in range(1, sf_index):
            new_config['SBF'][f'SF{idx}'] = self.config['SBF'][f'SF{idx}']
        for idx in range(sf_index, sf_count):
            new_config['SBF'][f'SF{idx}'] = self.config['SBF'][f'SF{idx + 1}']

        self.sf = new_sf
        self.set_config(new_config)

    def set_config(self, config):
        self.config = config
        self.name_index = self.get_name_index_dict()
        self.sf_names = self.get_sf_names()

    def add_sf(self, name, sf_to_add):
        sf_count = int(self.config['SBF']['SFCount'])
        self.config['SBF'][f'SF{sf_count + 1}'] = name
        self.config['SBF']['SFCount'] = str(sf_count + 1)  # add 1 to sf count
        self.sf = np.c_[self.sf, sf_to_add]  # add the column to the array

    def rename_sf(self, name, new_name):
        index = self.name_index[name]
        self.config['SBF'][f'SF{index + 1}'] = new_name

    def merge(self, sbf_data):
        if sbf_data.pc.shape[1] != 3:
            raise ValueError('[SbfData.merge] number of columns of pc shall be 3')

        if sbf_data.sf.shape[1] != self.sf.shape[1]:
            raise ValueError('[SbfData.merge] number of scalar fields shall be the same for merging')

        self.xyz = np.r_[self.xyz, sbf_data.xyz]
        self.sf = np.r_[self.sf, sbf_data.sf]

        print(f'[SbfData.merge] {len(sbf_data.pc)} points added, new total = {self.xyz.shape[0]}')

        self.config['SBF']['Points'] = str(self.xyz.shape[0])


def read_sbf(filename, verbose=True):
    return SbfData(filename, verbose=verbose)


def open_sbf(filename):
    return read_sbf_header(filename)
