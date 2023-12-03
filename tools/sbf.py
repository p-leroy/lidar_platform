import configparser
import logging
import os
import struct

import numpy as np

logger = logging.getLogger(__name__)


def shift_array(array, shift, config=None, debug=False):
    newArray = array.astype(float)
    # apply the shift read_bfe in the SBF file
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


def write_sbf(sbf, pc, sf=None, config=None, add_index=False, normals=None):
    head, tail = os.path.split(sbf)
    path_to_sbf = sbf
    path_to_sbf_data = sbf + '.data'
    n_points = pc.shape[0]
    if sf is not None:
        sf = sf.reshape(n_points, -1)
        SFCount = sf.shape[1]
    else:
        SFCount = 0

    # write .sbf
    Points = pc.shape[0]
    if config is None:
        dict_SF = {f'SF{k + 1}': f'{k + 1}' for k in range(SFCount)}
        config = configparser.ConfigParser()
        config.optionxform = str
        config['SBF'] = {'Points': str(Points),
                         'SFCount': str(SFCount),
                         'GlobalShift': '0., 0., 0.',
                         **dict_SF}
    else:
        # enforce the coherence of the number of points
        config['SBF']['Points'] = str(Points)
        config['SBF']['SFCount'] = str(SFCount)

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
    globalShift = eval(config['SBF']['GlobalShift'])
    pcOrig = pc - np.array(globalShift).reshape(1, -1)
    # compute sbf internal shift
    shift = np.mean(pcOrig, axis=0).astype(float)
    # build the array that will effectively be stored (32 bits float)
    a = np.zeros((Points, SFCount + 3)).astype('>f')
    a[:, :3] = (pcOrig - shift).astype('>f')
    if SFCount != 0:
        a[:, 3:] = sf.astype('>f')

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
    def __init__(self, filename):
        self.filename = filename
        self.pc = None
        self.sf = None
        self.config = None

        self.read_sbf(filename)  # set pc, sf and config

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
            array = np.fromfile(f, dtype='>f').reshape(Np, Ns + 3)
            shift = np.array((x_shift, y_shift, z_shift)).reshape(1, 3)

        # shift point cloud
        pc = shift_array(array[:, :3], shift, config)

        # get scalar fields if any
        if Ns != 0:
            sf = array[:, 3:]
        else:
            sf = None

        self.pc = pc
        self.sf = sf
        self.config = config

    def get_name_index_dict(self):
        dict_ = {self.config['SBF'][name]: int(name.split('SF')[1]) - 1
                 for name in self.config['SBF'] if len(name.split('SF')) == 2 and is_int(name.split('SF')[1])}
        return dict_

    def remove_sf(self, name, sf, config):
        name_index = self.get_name_index_dict(config)
        # remove the scalar field from the sf array
        index = name_index[name]
        new_sf = np.delete(sf, index, axis=1)
        # copy the configuration
        new_config = configparser.ConfigParser()
        new_config.optionxform = str
        new_config.read_dict(config)
        sf_index = index + 1
        sf_count = int(config['SBF']['SFCount'])
        # update the configuration
        new_config['SBF']['SFCount'] = str(sf_count - 1)  # decrease the counter of scalar fields
        new_config.remove_option('SBF', f'SF{sf_count}')  # remove the last option
        for idx in range(1, sf_index):
            new_config['SBF'][f'SF{idx}'] = config['SBF'][f'SF{idx}']
        for idx in range(sf_index, sf_count):
            new_config['SBF'][f'SF{idx}'] = config['SBF'][f'SF{idx + 1}']

        self.sf = new_sf
        self.config = new_config

    def add_sf(self, name, sf_to_add):
        sf_count = int(self.config['SBF']['SFCount'])
        self.config['SBF'][f'SF{sf_count + 1}'] = name
        self.config['SBF']['SFCount'] = str(sf_count + 1)  # add 1 to sf count
        self.sf = np.c_[self.sf, sf_to_add]  # add the clumn to the array

    def rename_sf(self, name, new_name):
        name_index = self.get_name_index_dict(self.config)
        index = name_index[name]
        self.config['SBF'][f'SF{index + 1}'] = new_name


def read(filename):
    return SbfData(filename)
