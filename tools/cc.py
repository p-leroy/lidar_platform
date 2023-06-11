# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:17:49 2021

@author: Paul Leroy
"""

import configparser
import logging
import os
import shutil
import struct

import numpy as np

from ..config.config import cc_custom, cc_std, cc_std_alt
from ..tools import misc

logger = logging.getLogger(__name__)

EXIT_FAILURE = 1
EXIT_SUCCESS = 0

#############################
# BUILD CLOUD COMPARE COMMAND
#############################


class CCCommand(list):
    def __init__(self, cc_exe, silent=True, fmt='SBF'):
        self.append(cc_exe)
        if silent:
            self.append('-SILENT')
        self.append('-NO_TIMESTAMP')
        self.append('-C_EXPORT_FMT')
        if fmt.lower() == 'laz':  # needed to export to laz
            self.append('LAS')
            self.append("-EXT")
            self.append("laz")
        else:
            self.append(fmt)

    def open_file(self, fullname, global_shift='AUTO', fwf=False):
        if not os.path.exists(fullname):
            raise FileNotFoundError(fullname)
        if fwf:
            self.append('-fwf_o')
        else:
            self.append('-o')
        if global_shift is not None:
            self.append('-GLOBAL_SHIFT')
            if global_shift == 'AUTO' or global_shift == 'FIRST':
                self.append(global_shift)
            elif type(global_shift) is tuple or type(global_shift) is list:
                x, y, z = global_shift
                self.append(str(x))
                self.append(str(y))
                self.append(str(z))
            else:
                raise ValueError('invalid value for global_shit')
        self.append(fullname)


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class CloudCompareError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self):
        pass


def format_name(in_, name_):
    # generating a cloud_file full name for the subprocess call can be tricky
    # especially when the path contains whitespaces...
    # handling all these whitespaces is really tricky...
    # sigh... (read CC command line help, option -FILE)
    normpath = os.path.normpath(os.path.join(in_, name_))
    list_ = [f'"{item}"' if ' ' in item else item for item in normpath.split('\\')]
    if ':' in list_[0]:
        new_name = '/'.join(list_)  # beurk
    else:
        new_name = os.path.join(*list_)
    return new_name


def cloud_exists(cloud, verbose=False):
    head, tail = os.path.split(cloud)
    if os.path.exists(cloud):
        if verbose is True: 
            logger.info(f'cloud {tail} exists')
        return True
    else:
        logger.error(f'cloud {tail} does not exist')
        raise Error(f'cloud {tail} does not exist')


def copy_cloud(cloud, out):
    # copy the cloud file to out
    # /!\ with sbf files, one shall copy .sbf and .sbf.data
    head, tail = os.path.split(cloud)
    root, ext = os.path.splitext(cloud)
    logger.info(f'copy {tail} to output directory')
    dst = shutil.copy(cloud, out)
    if ext == '.sbf':
        src = cloud + '.data'
        if os.path.exists(src):
            logger.info('copy .sbf.data to output directory')
            shutil.copy(src, out)
    return dst


def move_cloud(cloud, odir):
    # move the cloud to out
    # /!\ with sbf files, one shall copy .sbf and .sbf.data
    head, tail = os.path.split(cloud)
    root, ext = os.path.splitext(cloud)
    logger.info(f'move {tail} to output directory')
    if os.path.isdir(odir):
        out = os.path.join(odir, tail)
    else:
        out = odir
    dst = shutil.move(cloud, out)
    if ext == '.sbf':
        src = cloud + '.data'
        if os.path.exists(src):
            logger.info('move .sbf.data to output directory')
            shutil.move(src, out + '.data')
    return dst


def merge(files, fmt='sbf',
          silent=True, debug=False, global_shift='AUTO', cc=cc_std_alt):

    if len(files) == 1 or files is None:
        print("[cc.merge] only one file in parameter 'files', this is quite unexpected!")
        return None

    cmd = CCCommand(cc, silent=silent, fmt=fmt)
    if global_shift == 'FIRST':
        raise "'FIRST' is not a valid option, the default is 'AUTO' or pass a valid global shift 3-tuple"
    elif global_shift =='AUTO':
        print("[cc.merge] WARNING be careful when using 'AUTO' if the resulting shifted coordinates are still large")
        cmd.open_file(files[0], global_shift='AUTO')
        for file in files[1:]:
            cmd.open_file(file, global_shift='FIRST')
    else:
        for file in files:
            cmd.open_file(file, global_shift=global_shift)
    cmd.append('-MERGE_CLOUDS')

    misc.run(cmd, verbose=debug)

    root, ext = os.path.splitext(files[0])
    return root + f'_MERGED.{fmt.lower()}'


def sf_interp_and_merge(src, dst, index, global_shift, silent=True, debug=False, cc=cc_custom, export_fmt='sbf'):
    x, y, z = global_shift
    args = ''
    if silent is True:
        args += ' -SILENT -NO_TIMESTAMP'
    else:
        args += ' -NO_TIMESTAMP'
    args += f' -C_EXPORT_FMT {export_fmt}'
    args += f' -o -GLOBAL_SHIFT {x} {y} {z} {src}'
    args += f' -o -GLOBAL_SHIFT {x} {y} {z} {dst}'
    args += f' -SF_INTERP {index}'  # interpolate scalar field from src to dst
    args += ' -MERGE_CLOUDS'

    misc.run(cc + args, verbose=debug)
    root, ext = os.path.splitext(src)
    return root + f'_MERGED.{export_fmt}'


def density(pc, radius, density_type,
            silent=True, debug=False, global_shift='AUTO'):
    """ Compute the density on a cloud

    :param pc:
    :param radius:
    :param density_type: type can be KNN SURFACE VOLUME
    :param silent:
    :param debug:
    :param global_shift:
    :return:
    """

    cmd = CCCommand(cc_std_alt, silent=silent, fmt='SBF')
    cmd.append('-SAVE_CLOUDS')
    cmd.open_file(pc, global_shift=global_shift)
    cmd.append('-REMOVE_ALL_SFS')
    cmd.append('-DENSITY')
    cmd.append(str(radius))
    cmd.append('-TYPE')
    cmd.append(density_type)
    misc.run(cmd, verbose=debug)

    root, ext = os.path.splitext(pc)
    return root + '_DENSITY.sbf'

#########################################################
#  3DMASC KEEP_ATTRIBUTES / ONLY_FEATURES / SKIP_FEATURES
#########################################################


def q3dmasc_get_labels(training_file):
    # if 'core_points:' is defined, the main cloud is the cloud defined by core_points
    # if not, the main cloud is the first occurence of 'clouds:'
    with open(training_file, 'r') as f:
        clouds = []
        core_points = None
        for line in f.readlines():
            if line[0] == '#':
                pass
            else:
                if 'CLOUD:' in line.upper():
                    clouds.append(line[7:].strip().split('=')[0])
                if 'CORE_POINTS:' in line.upper():
                    core_points = line[12:].strip().split('_')[0]
        if core_points is not None:
            main_cloud = core_points
        else:
            main_cloud = clouds[0]
        return main_cloud, clouds


def q3dmasc_only_features(clouds, training_file,
                          silent=True, verbose=False, global_shift='AUTO', cc_exe=cc_custom):
    """Command line call to 3DMASC with the only_features option.

    In command line, the clouds to load are not read in the parameter file, you have to specify them in the call
    and you also have to associate each label to a number, the number representing the order in which the clouds
    have been loaded

    :param clouds: a list of cloud paths or a unique cloud path
    :param training_file: a 3DMASC parameter file
    :param silent:
    :param verbose:
    :param global_shift:
    :param cc_exe:
    :return: the name of the output file
    """

    main_label, labels = q3dmasc_get_labels(training_file)  # get cloud labels from the parameter file

    cmd = CCCommand(cc_exe, silent=silent, fmt='SBF')  # create the command
    cloud_dict = {}  # will be used to generate the name of the output file
    if type(clouds) == list:
        for i, cloud in enumerate(clouds):
            cmd.open_file(cloud, global_shift=global_shift)
            cloud_dict[labels[i]] = cloud
    else:
        cmd.open_file(clouds, global_shift=global_shift)

    cmd.append('-3DMASC_CLASSIFY')
    cmd.append('-ONLY_FEATURES')
    cmd.append(training_file)

    # generate the string where roles are associated with open clouds, e.g. 'pc1=1 pc2=2'
    role_association = ' '.join([f'{label}={i + 1}' for i, label in enumerate(labels)])
    cmd.append(role_association)

    misc.run(cmd, verbose=verbose)

    root, ext = os.path.splitext(cloud_dict[main_label])
    return root + '_WITH_FEATURES.sbf'


################
# Best Fit Plane
################


def best_fit_plane(cloud, debug=False):
    
    cloud_exists(cloud)
    
    args = ''
    args += ' -SILENT -NO_TIMESTAMP'
    args += ' -o ' + cloud
    args += ' -BEST_FIT_PLANE '
    
    misc.run(cc_custom + args, verbose=debug)
    
    outputs = (os.path.splitext(cloud)[0] + '_BEST_FIT_PLANE.bin',
               os.path.splitext(cloud)[0] + '_BEST_FIT_PLANE_INFO.txt')
    
    return outputs


def get_orientation_matrix(filename):
    with open(filename) as f:
        matrix = np.genfromtxt(f, delimiter=' ', skip_header=5)
    return matrix

######
# M3C2
######


def m3c2(pc1, pc2, params, core=None, fmt='SBF',
         silent=True, debug=False, global_shift='AUTO', cc=cc_std_alt):

    cmd = CCCommand(cc, silent=silent, fmt=fmt)
    if global_shift == 'FIRST':
        raise "'FIRST' is not a valid option, the default is 'AUTO' or pass a valid global shift 3-tuple"
    elif global_shift =='AUTO':
        print("[cc.m3c2] WARNING be careful when using 'AUTO' if the resulting shifted coordinates are still large")
        cmd.open_file(pc1, global_shift='AUTO')
        cmd.open_file(pc2, global_shift='FIRST')
        if core is not None:
            cmd.open_file(core, global_shift='FIRST')
    else:
        cmd.open_file(pc1, global_shift=global_shift)
        cmd.open_file(pc2, global_shift=global_shift)
        if core is not None:
            cmd.open_file(core, global_shift=global_shift)
    cmd.append("-M3C2")
    cmd.append(params)

    misc.run(cmd, verbose=debug)

    root1, ext1 = os.path.splitext(pc1)
    results = root1 + f'_M3C2.{fmt.lower()}'
    return results

#######
# OTHER
#######


def drop_global_shift(cloud, silent=True):
    args = ''
    if silent is True:
        args += ' -SILENT -NO_TIMESTAMP'
    else:
        args += ' -NO_TIMESTAMP'
    args += ' -o ' + cloud
    args += ' -DROP_GLOBAL_SHIFT -SAVE_CLOUDS'
    ret = misc.run(cc_custom + args)
    if ret == EXIT_FAILURE:
        raise CloudCompareError
    return ret


def remove_scalar_fields(cloud, silent=True):
    args = ''
    if silent is True:
        args += ' -SILENT -NO_TIMESTAMP'
    else:
        args += ' -NO_TIMESTAMP'
    args += ' -o ' + cloud
    args += ' -REMOVE_ALL_SFS -SAVE_CLOUDS'
    misc.run(cc_custom + args)


def rasterize(cloud, spacing, ext='_RASTER', proj='AVG', fmt='SBF',
              silent=True, debug=False, global_shift='AUTO', cc=cc_std_alt):
    cloud_exists(cloud)
    if not os.path.exists(cloud):
        raise FileNotFoundError

    cmd = CCCommand(cc_std_alt, silent=silent, fmt=fmt)
    cmd.open_file(cloud, global_shift=global_shift)
    cmd.append('-RASTERIZE')
    cmd.append('-GRID_STEP')
    cmd.append(str(spacing))
    cmd.append('-PROJ')
    cmd.append(proj)

    misc.run(cmd, verbose=debug)
    
    return os.path.splitext(cloud)[0] + ext + f'.{fmt.lower()}'

##########
#  ICPM3C2
##########


def icpm3c2(pc1, pc2, params, core=None, silent=True, fmt='BIN', debug=False):
    cloud_exists(pc1, verbose=False)
    cloud_exists(pc2, verbose=False)
    args = ''
    if silent is True:
        args += ' -SILENT -NO_TIMESTAMP'
    else:
        args += ' -NO_TIMESTAMP'
    if fmt is None:
        pass
    else:
        args += f' -C_EXPORT_FMT {fmt}'
    args += ' -o -GLOBAL_SHIFT FIRST ' + pc1
    args += ' -o -GLOBAL_SHIFT FIRST ' + pc2
    if core is not None:
        args += ' -o -GLOBAL_SHIFT FIRST ' + core
    args += ' -ICPM3C2 ' + params
    cmd = cc_custom + args
    if debug is True:
        logging.info(cmd)
    ret = misc.run(cmd, verbose=debug)
    if ret == EXIT_FAILURE:
        raise CloudCompareError
    # extracting rootname of the fixed point cloud Q
    if fmt == 'SBF':
        ext = 'sbf'
    elif fmt == 'BIN':
        ext = 'bin'
    elif fmt == 'ASC':
        ext = 'asc'
    else:
        ext = 'bin'
    head2, tail2 = os.path.split(pc2)
    root2, ext2 = os.path.splitext(tail2)
    results = os.path.join(head2, root2 + f'_ICPM3C2.{ext}')
    return results

#################
#  TO BIN, TO SBF
#################


def to_bin(fullname, debug=False, shift=None, cc=cc_std):
    root, ext = os.path.splitext(fullname)
    if os.path.exists(fullname):
        args = ''
        if debug==False:
            args += ' -SILENT -NO_TIMESTAMP'
        else:
            args += ' -NO_TIMESTAMP'
        args += ' -C_EXPORT_FMT BIN'
        if shift is not None:
            x, y, z = shift
            args += f' -o -GLOBAL_SHIFT {x} {y} {z} ' + fullname
        else:
            args += ' -o ' + fullname
        args += ' -SAVE_CLOUDS'
        print(f'cc {args}')
        ret = misc.run(cc + args, verbose=debug)
        return ret
    else:
        print(f'error, {fullname} does not exist')
        return -1


def all_to_bin(dir_, shift, debug=False):
    list_ = os.listdir(dir_)
    for name in list_:
        path = os.path.join(dir_, name)
        if os.path.isfile(path):
            if os.path.splitext(path)[-1] == '.laz':
                to_bin(path, debug=debug, shift=shift)


def to_laz(fullname, remove=False, save_clouds='SAVE_CLOUDS',
           silent=True, debug=False, global_shift='AUTO', cc_exe=cc_std_alt):
    """

    :param fullname:
    :param remove:
    :param save_clouds: SAVE_CLOUDS or FWF_SAVE_CLOUDS
    :param silent:
    :param debug:
    :param global_shift:
    :param cc_exe:
    :return:
    """

    if not os.path.exists(fullname):
        raise FileNotFoundError

    root, ext = os.path.splitext(fullname)
    if ext == '.laz':  # nothing to do, simply return the name
        return fullname

    cmd = CCCommand(cc_exe, silent=silent, fmt='LAZ')
    cmd.open_file(fullname, global_shift=global_shift)
    cmd.append(f'-{save_clouds}')
    misc.run(cmd, verbose=debug)

    if remove:
        print(f'remove {fullname}')
        os.remove(fullname)
        if ext == '.sbf':
            to_remove = fullname + '.data'
            print(f'remove {to_remove}')
            os.remove(to_remove)
    return os.path.splitext(fullname)[0] + '.laz'


def to_sbf(fullname,
           silent=True, debug=False, global_shift='AUTO', cc_exe=cc_std_alt, fwf=False):

    cmd = CCCommand(cc_exe, silent=silent, fmt='SBF')
    cmd.open_file(fullname, global_shift=global_shift, fwf=fwf)

    root, ext = os.path.splitext(fullname)
    if ext == '.sbf':  # nothing to do, simply return the name
        out = fullname
    else:
        cmd.append('-SAVE_CLOUDS')
        misc.run(cmd, verbose=debug)
        out = os.path.splitext(fullname)[0] + '.sbf'
    return out


##############
#  SUBSAMPLING
##############


def ss(fullname, algorithm='OCTREE', parameter=8, odir=None, fmt='SBF',
       silent=True, debug=False, global_shift='AUTO', cc_exe=cc_std_alt):
    """
    Use CloudCompare to subsample a cloud.

    :param fullname: the full name of the cloud to subsample
    :param algorithm: RANDOM SPATIAL OCTREE
    :param parameter: number of points / distance between points / subdivision level
    :param odir: output directory
    :param fmt: output format
    :param silent: use CloudCompare in silent mode
    :param debug:
    :param global_shift:
    :param cc_exe: CloudCompare executable
    :return: the name of the output file
    """

    print(f'[cc.ss] subsample {fullname}')

    cmd = CCCommand(cc_exe, silent=silent, fmt=fmt)
    cmd.open_file(fullname, global_shift=global_shift)

    cmd.append('-SS')
    cmd.append(algorithm)
    cmd.append(str(parameter))
    ret = misc.run(cmd, verbose=debug)

    root, ext = os.path.splitext(fullname)
    os.makedirs(odir, exist_ok=True)
    if algorithm == 'OCTREE':
        out = root + f'_OCTREE_LEVEL_{parameter}_SUBSAMPLED.{fmt.lower()}'
    elif algorithm == 'SPATIAL':
        out = root + f'_SPATIAL_SUBSAMPLED.{fmt.lower()}'
    elif algorithm == 'RANDOM':
        out = root + f'_RANDOM_SUBSAMPLED.{fmt.lower()}'

    if odir:
        head, tail = os.path.split(out)
        dst = os.path.join(odir, tail)
        shutil.move(out, dst)
        if fmt == 'SBF':
            dst_data = os.path.join(odir, tail + '.data')
            shutil.move(out + '.data', dst_data)
        out = dst
        return out

#######################
#  CLOUD TRANSFORMATION
#######################


def get_inverse_transformation(transformation):
    R = transformation[:3, :3]
    T = transformation[:3:, 3]
    inv = np.zeros((4, 4))
    inv[3, 3] = 1
    inv[:3, :3] = R.T
    inv[:3:, 3] = -R.T @ T
    return inv


def save_trans(out, R, T):
    transformation = np.zeros((4, 4))
    transformation[:3, :3] = R
    transformation[:3, 3, None] = T
    transformation[3, 3] = 1
    np.savetxt(out, transformation, fmt='%.8f')
    logger.debug(f'{transformation} saved')


def apply_trans_alt(cloudfile, transformation):
    args = ''
    args += ' -SILENT -NO_TIMESTAMP'
    args += ' -o ' + cloudfile
    args += ' -APPLY_TRANS ' + transformation
    ret = misc.run(cc_custom + args)
    if ret == EXIT_FAILURE:
        raise CloudCompareError
    root, ext = os.path.splitext(cloudfile)
    return root + '_TRANSFORMED.bin'


def apply_transformation(cloudfile, transformation, fmt='SBF',
                         global_shift=None, silent=True, debug=False):
    """
    Transform a point cloud using CloudCompare using a transformation specified in a text file.

    :param cloudfile:
    :param transformation:
    :param fmt:
    :param global_shift:
    :param silent:
    :param debug:
    :return:
    """

    print(f'[cc.apply_transformation] apply transformation to {cloudfile}')
    if not os.path.exists(cloudfile):
        raise FileNotFoundError

    root, ext = os.path.splitext(cloudfile)
    out = root + f'_TRANSFORMED.{fmt.lower()}'
    level = logger.getEffectiveLevel()
    if debug is True:
        logger.setLevel(logging.DEBUG)

    cmd = CCCommand(cc_std_alt, silent=silent, fmt=fmt)
    cmd.open_file(cloudfile, global_shift=global_shift)

    cmd.append('-APPLY_TRANS')
    cmd.append(transformation)
    misc.run(cmd, verbose=debug)

    return out


def transform_cloud(cloud, R, T, shift=None, silent=True, debug=False):
    cloud_exists(cloud)
    head, tail = os.path.split(cloud)
    cloud_trans = cloud # the transformation will overwrite the cloud
    transformation = os.path.join(head, 'transformation.txt')
    # Create the matrix file to be used by CloudCompare to transform the cloud
    save_trans(transformation, R, T)
    # Transform the cloud
    logger.info(f'[CC] transformation of {tail}')
    apply_transformation(cloud, transformation, cloud_trans, silent=silent, debug=debug, shift=shift)

###################
#  BIN READ / WRITE
###################


def get_from_bin(bin_):
    with open(bin_, 'rb') as f:
        bytes_ = f.read(4)
        # 'I' unsigned int / integer / 4
        for k in range(3):
            print(chr(bytes_[k]))

#################
#  SBF READ/WRITE
#################


def is_int(str_):
    try:
        int(str_)
        return True
    except ValueError:
        return False


def get_name_index_dict(config):
    dict_ = {config['SBF'][name]: int(name.split('SF')[1]) - 1
             for name in config['SBF'] if len(name.split('SF')) == 2 and is_int(name.split('SF')[1])}
    return dict_


def remove_sf(name, sf, config):
    name_index = get_name_index_dict(config)
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
    return new_sf, new_config


def add_sf(name, sf, config, sf_to_add):
    sf_count = int(config['SBF']['SFCount'])
    config['SBF'][f'SF{sf_count + 1}'] = name
    config['SBF']['SFCount'] = str(sf_count + 1)  # add 1 to sf count
    sf = np.c_[sf, sf_to_add]  # add the clumn to the array
    return sf


def rename_sf(name, new_name, config):
    name_index = get_name_index_dict(config)
    index = name_index[name]
    config['SBF'][f'SF{index + 1}'] = new_name


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


def read_sbf(sbf, verbose=False):

    config = read_sbf_header(sbf, verbose=verbose)  # READ .sbf header
    
    ################
    # READ .sbf.data
    # be careful, sys.byteorder is probably 'little' (different from Cloud Compare)
    sbf_data = sbf + '.data'
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
        array = np.fromfile(f, dtype='>f').reshape(Np, Ns+3)
        shift = np.array((x_shift, y_shift, z_shift)).reshape(1, 3)
        
    # shift point cloud
    pc = shift_array(array[:, :3], shift, config)
    
    # get scalar fields if any
    if Ns != 0:
        sf = array[:, 3:]
    else:
        sf = None
        
    return pc, sf, config


def write_sbf(sbf, pc, sf, config=None, add_index=False, normals=None):
    head, tail = os.path.split(sbf)
    path_to_sbf = sbf
    path_to_sbf_data = sbf + '.data'
    if sf is not None:
        SFCount = sf.shape[1]
    else:
        SFCount = 0
    
    # write .sbf
    Points = pc.shape[0] 
    if config is None:
        dict_SF = {f'SF{k+1}':f'{k+1}' for k in range(SFCount)}
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
        config['SBF'][f'SF{SFCount+1}'] = 'Nx'
        config['SBF'][f'SF{SFCount+2}'] = 'Ny'
        config['SBF'][f'SF{SFCount+3}'] = 'Nz'
    
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
        sbf_data.write(bytes(63-36+1))
        sbf_data.write(a)
        
##########
# C2C_DIST
##########


def c2c_dist(compared, reference, max_dist=None, split_XYZ=False, odir=None, export_fmt='SBF',
             silent=True, debug=False, global_shift='AUTO', cc_exe=cc_std_alt):

    cmd = CCCommand(cc_exe, silent=silent, fmt='SBF')  # create the command
    cmd.open_file(compared, global_shift=global_shift)
    cmd.open_file(reference, global_shift=global_shift)

    cmd.append('-c2c_dist')

    if split_XYZ is True:
        cmd.append('-SPLIT_XYZ')
    if max_dist:
        cmd.append('-MAX_DIST')
        cmd.append(str(max_dist))

    misc.run(cmd, verbose=debug)

    root, ext = os.path.splitext(compared)
    if max_dist:
        output = root + f'_C2C_DIST_MAX_DIST_{max_dist}.sbf'
    else:
        output = root + '_C2C_DIST.sbf'
    head, tail = os.path.split(output)

    # move the result if odir has been set
    if os.path.exists(odir) and odir is not None:
        overlap = os.path.join(odir, tail)
        shutil.move(output, overlap)
        if export_fmt.lower() == 'sbf':  # move .sbf.data also in cas of sbf export format
            shutil.move(output + '.data', overlap + '.data')
        output = overlap
    
    return output


def closest_point_set(compared, reference, silent=True, debug = False):
    compRoot, compExt = os.path.splitext(compared)
    compHead, compTail = os.path.split(compared)
    refRoot, refExt = os.path.splitext(reference)
    args = ''
    if silent is True:
        args += ' -SILENT -NO_TIMESTAMP'
    else:
        args += ' -NO_TIMESTAMP'
    args += ' -C_EXPORT_FMT SBF'
    if compExt == '.sbf':
        args += f' -o -GLOBAL_SHIFT FIRST {compared}'
    else:
        args += f' -o {compared}'
    if refExt == '.sbf':
        args += f' -o -GLOBAL_SHIFT FIRST {reference}'
    else:
        args += f' -o {reference}'
    args += ' -CLOSEST_POINT_SET'

    compBase = os.path.splitext(os.path.split(compared)[1])[0]
    refBase = os.path.splitext(os.path.split(reference)[1])[0]

    misc.run(cc_custom + args, verbose=debug)
    
    return os.path.join(compHead, f'[{refBase}]_CPSet({compBase}).sbf')

#####
# ICP
#####


def icp(compared, reference,
        overlap=None,
        random_sampling_limit=None, 
        farthest_removal=False,
        iter_=None,
        silent=True, debug=False):
    compRoot, compExt = os.path.splitext(compared)
    refRoot, refExt = os.path.splitext(reference)
    args = ''
    if silent is True:
        args += ' -SILENT -NO_TIMESTAMP'
    else:
        args += ' -NO_TIMESTAMP'
    args += ' -C_EXPORT_FMT SBF'
    if compExt == '.sbf':
        args += f' -o -GLOBAL_SHIFT FIRST {compared}'
    else:
        args += f' -o {compared}'
    if refExt == '.sbf':
        args += f' -o -GLOBAL_SHIFT FIRST {reference}'
    else:
        args += f' -o {reference}'
    args += ' -ICP'
    if overlap is not None:
        args += f' -OVERLAP {overlap}'
    if random_sampling_limit is not None:
        args += f' -RANDOM_SAMPLING_LIMIT {random_sampling_limit}'
    if farthest_removal is True:
        args += ' -FARTHEST_REMOVAL'
    if iter_ is not None:
        args += f' -ITER {iter_}'

    print(f'cc {args}')
    misc.run(cc_custom + args, verbose=debug)
    
    out = os.path.join(os.getcwd(), 'registration_trace_log.csv')
    return out
