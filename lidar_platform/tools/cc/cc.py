# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:17:49 2021

@author: Paul Leroy
"""

import logging
import os
import shutil

import numpy as np

from ...config.config import cc_custom, cc_std, cc_exe
from .. import misc

from .CCCommand import CCCommand

logger = logging.getLogger(__name__)

EXIT_FAILURE = 1
EXIT_SUCCESS = 0


#############################
# BUILD CLOUD COMPARE COMMAND
#############################

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
    # sigh... (read_bfe CC command line help, option -FILE)
    normpath = os.path.normpath(os.path.join(in_, name_))
    list_ = [f'"{item}"' if ' ' in item else item for item in normpath.split('\\')]
    if ':' in list_[0]:
        new_name = '/'.join(list_)  # bad
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
    print(f'move {tail} to {odir}')
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
          silent=True, debug=False, global_shift='AUTO', cc=cc_exe):

    if len(files) == 1 or files is None:
        print("[cc.merge] only one file in parameter 'files', this is quite unexpected!")
        return None

    cmd = CCCommand(cc, silent=silent, fmt=fmt, auto_save='on')
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


def sf_interp_and_merge(src, dst, index, global_shift,
                        silent=True, debug=False, cc=cc_custom, export_fmt='sbf'):
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
            silent=True, verbose=False, global_shift='AUTO'):
    """ Compute the density on a cloud

    :param pc:
    :param radius:
    :param density_type: type can be KNN SURFACE VOLUME
    :param silent:
    :param verbose:
    :param global_shift:
    :return:
    """

    root, ext = os.path.splitext(pc)
    out = root + '_DENSITY.sbf'

    cmd = CCCommand(cc_exe, silent=silent, fmt='SBF')
    cmd.open_file(pc, global_shift=global_shift)
    cmd.append('-DENSITY')
    cmd.append(str(radius))
    cmd.append('-TYPE')
    cmd.append(density_type)
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd, verbose=verbose)

    return out


######
# M3C2
######


def m3c2(pc1, pc2, params, core=None, fmt='SBF',
         silent=True, debug=False, global_shift='AUTO', cc=cc_exe):

    if not os.path.exists(params):
        raise FileNotFoundError(params)

    cmd = CCCommand(cc, silent=silent, auto_save='ON', fmt=fmt)

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

    if not os.path.exists(params):
        raise FileNotFoundError(params)
    cmd.append(params)

    misc.run(cmd, verbose=debug)

    root1, ext1 = os.path.splitext(pc1)
    results = root1 + f'_M3C2.{fmt.lower()}'
    return results


##########
#  ICPM3C2
##########


def icpm3c2(pc1, pc2, params, core=None, silent=True, fmt='BIN', verbose=False, cc_exe=cc_custom,
            global_shift=None):

    cmd = CCCommand(cc_exe, silent=silent, fmt=fmt)
    if global_shift is None:
        cmd.open_file(pc1, global_shift='AUTO')
        cmd.open_file(pc2, global_shift='FIRST')
    else:
        cmd.open_file(pc1, global_shift=global_shift)
        cmd.open_file(pc2, global_shift=global_shift)
    if core is not None:
        cmd.open_file(core)
    cmd.extend(['-ICPM3C2', params])

    if verbose is True:
        logging.info(cmd)
    # ret = misc.run(cmd, verbose=verbose)
    # if ret == EXIT_FAILURE:
    #     raise CloudCompareError
    misc.run(cmd, verbose=verbose)

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


#########################################################
#  3DMASC KEEP_ATTRIBUTES / ONLY_FEATURES / SKIP_FEATURES
#########################################################


def q3dmasc_get_labels(training_file):
    # if 'core_points:' is defined, the main cloud is the cloud defined by core_points
    # if not, the main cloud is the first occurrence of 'cloud:'
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


def q3dmasc(clouds, training_file, only_features=False, keep_attributes=False,
            silent=True, verbose=False, global_shift='AUTO', cc_exe=cc_exe, fmt='sbf'):
    """Command line call to 3DMASC with the only_features option.

    The Python function generates the command line for you. Some details about the command are given below.
    For the Python call, you simply have to specify a list of clouds which respects the order of appearance of the roles
    in the parameter file.
    clouds[0] will be associated to the first read label
    clouds[1] to the second one
    and so on...

    Details about the call in command line:
    In command line, the clouds to load are not read in the parameter file, you have to specify them in the call,
    and you also have to associate each label to a number, the number representing the order in which the clouds
    have been open (-O option in the command line).

    :param clouds: a list of cloud paths or a unique cloud path
    :param training_file: a 3DMASC parameter file or a classifier, depending on the options
    :param only_features: stop after the features computation, no training, no classification [True / False]
    :param keep_attributes: keep attributes after completion
    :param silent: call CloudCompare in silent mode [True / False]
    :param verbose: verbose mode for the command line
    :param global_shift: global_shifht for the opening of point clouds by CloudCompare
    :param cc_exe: the CloudCompare executable
    :return: the name of the output file
    """

    main_label, labels = q3dmasc_get_labels(training_file)  # get cloud labels from the parameter file
    print(f'[q3dmasq] labels read in the parameter file {labels}, compute features on {main_label}')

    cmd = CCCommand(cc_exe, silent=silent, fmt=fmt, auto_save='ON')  # create the command
    cloud_dict = {}  # will be used to generate the name of the output file
    if type(clouds) is list or type(clouds) is tuple:
        for i, cloud in enumerate(clouds):
            cmd.open_file(cloud, global_shift=global_shift)
            cloud_dict[labels[i]] = cloud
    else:
        cmd.open_file(clouds, global_shift=global_shift)

    cmd.append('-3DMASC_CLASSIFY')
    if only_features:
        cmd.append('-ONLY_FEATURES')
    else:
        if keep_attributes:
            cmd.append('-KEEP_ATTRIBUTES')
    cmd.append(training_file)

    # generate the string where roles are associated with open clouds, e.g. 'pc1=1 pc2=2'
    role_association = ' '.join([f'{label}={i + 1}' for i, label in enumerate(labels)])
    cmd.append(role_association)

    # remove expected output file to avoid strange effects when writing in an existing sbf file
    root, ext = os.path.splitext(cloud_dict[main_label])
    if only_features:
        output_name = root + '_WITH_FEATURES.' + fmt.lower()
    else:
        output_name = root + '_CLASSIFIED.' + fmt.lower()
    if os.path.exists(output_name):
        print(f'[q3dmasc] remove existing output file before 3DMASC call: {output_name}')
        os.remove(output_name)

    misc.run(cmd, verbose=verbose)

    return output_name

###############
# FULL WAVEFORM
###############


def compress_fwf(cloud, in_place=True,
              silent=True, debug=False, global_shift='AUTO', cc=cc_exe):

    if not os.path.exists(cloud):
        raise FileNotFoundError

    print(f'[compress_fwf] {cloud}')

    cmd = CCCommand(cc, silent=silent, fmt='LAS')
    cmd.open_file(cloud, global_shift=global_shift)
    cmd.append('-COMPRESS_FWF')
    if in_place:
        out = cloud
    else:
        out = os.path.splitext(cloud)[0] + '_compressed.laz'
    cmd.extend(['-SAVE_CLOUDS', 'file', out])

    misc.run(cmd, verbose=debug)

    return out


def fwf_peaks(cloud, ini,
              silent=True, debug=False, global_shift='AUTO', cc=cc_exe):

    if not os.path.exists(cloud):
        raise FileNotFoundError

    print(f'[fwf_ortho] {cloud}')

    out = os.path.splitext(cloud)[0] + '_fwf_ortho.laz'

    cmd = CCCommand(cc, silent=silent, fmt='LAS')
    cmd.open_file(cloud, global_shift=global_shift)
    cmd.extend(['-FWF_PEAKS', ini])

    cmd.extend(['-SAVE_CLOUDS', 'file', out])

    misc.run(cmd, verbose=debug)

    return out


def fwf_ortho(cloud, ini,
              silent=True, debug=False, global_shift='AUTO', cc=cc_exe):

    if not os.path.exists(cloud):
        raise FileNotFoundError

    print(f'[fwf_ortho] {cloud}')

    head, tail = os.path.split(cloud)
    root, ext = os.path.splitext(tail)
    odir = os.path.join(head, 'fwf_ortho')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '_fwf_ortho.laz')

    cmd = CCCommand(cc, silent=silent, fmt='LAS')
    cmd.open_file(cloud, global_shift=global_shift)
    cmd.extend(['-FWF_ORTHO', ini])
    cmd.extend(['-SAVE_CLOUDS', 'file', out])

    misc.run(cmd, verbose=debug)

    return out


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


def remove_all_scalar_fields(cloud, silent=True):
    args = ''
    if silent is True:
        args += ' -SILENT -NO_TIMESTAMP'
    else:
        args += ' -NO_TIMESTAMP'
    args += ' -o ' + cloud
    args += ' -REMOVE_ALL_SFS -SAVE_CLOUDS'
    misc.run(cc_custom + args)


def remove_scalar_fields(file, scalar_fields, silent=True):
    root, ext = os.path.splitext(file)
    cmd = CCCommand(cc_exe, silent=silent, auto_save='OFF', fmt=f'{ext[1:]}')
    cmd.open_file(file)
    for scalar_field in scalar_fields:
        cmd.append('-REMOVE_SF')
        cmd.append(scalar_field)
    cmd.append('-SAVE_CLOUDS')
    misc.run(cmd)


def rasterize(cloud, spacing, suffix='_RASTER', proj='AVG', fmt='SBF',
              silent=True, verbose=False, global_shift='AUTO', cc=cc_exe,
              resample=False):
    """

    Parameters
    ----------
    cloud
    spacing
    suffix
    proj : str
        MIN AVG MAX
    fmt
    silent
    verbose
    global_shift
    cc
    resample : bool
         to resample the input cloud

    Returns
    -------

    """

    cloud_exists(cloud)
    if not os.path.exists(cloud):
        raise FileNotFoundError

    cmd = CCCommand(cc, silent=silent, fmt=fmt)
    cmd.open_file(cloud, global_shift=global_shift)
    cmd.extend(['-RASTERIZE', '-GRID_STEP', str(spacing), '-PROJ', proj])

    if resample is True:
        cmd.append('-RESAMPLE')

    out = os.path.splitext(cloud)[0] + suffix + f'.{fmt.lower()}'
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd, verbose=verbose)
    
    return out


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


def to_laz(fullname, remove=False, silent=True, debug=False, global_shift='AUTO', cc_exe=cc_exe):
    """

    :param fullname:
    :param remove:
    :param silent:
    :param debug:
    :param global_shift:
    :param cc_exe:
    :return:
    """

    print(f'[to_laz] {fullname}')

    if not os.path.exists(fullname):
        raise FileNotFoundError

    root, ext = os.path.splitext(fullname)
    if ext == '.laz':  # nothing to do, simply return the name
        return fullname
    out = root + '.laz'

    cmd = CCCommand(cc_exe, silent=silent, fmt='LAS')
    cmd.open_file(fullname, global_shift=global_shift)
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])
    misc.run(cmd, verbose=debug)

    if remove:
        print(f'remove {fullname}')
        os.remove(fullname)
        if ext == '.sbf':
            to_remove = fullname + '.data'
            print(f'remove {to_remove}')
            os.remove(to_remove)

    return out


def to_sbf(fullname,
           silent=True, debug=False, global_shift='AUTO', cc_exe=cc_exe, fwf=False):

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


def ss(fullname, method='OCTREE', parameter=8, odir=None, fmt='SBF',
       silent=True, verbose=False, global_shift='AUTO', cc_exe=cc_exe):
    """
    Use CloudCompare to subsample a cloud.

    :param fullname: the full name of the cloud to subsample
    :param algorithm: RANDOM SPATIAL OCTREE
    :param parameter: number of points / distance between points / subdivision level
    :param odir: output directory
    :param fmt: output format
    :param silent: use CloudCompare in silent mode
    :param verbose:
    :param global_shift:
    :param cc_exe: CloudCompare executable
    :return: the name of the output file
    """

    print(f'[cc.ss] subsample {fullname}')

    if method not in ('RANDOM', 'SPATIAL', 'OCTREE'):
        raise ValueError(f'Unknown method: {method}')

    cmd = CCCommand(cc_exe, silent=silent, fmt=fmt)
    cmd.open_file(fullname, global_shift=global_shift)

    cmd.extend(['-SS', method, str(parameter)])

    root = os.path.splitext(fullname)[0]
    out = root + f'_{method}_SUBSAMPLED.{fmt.lower()}'
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    ret = misc.run(cmd, verbose=verbose)

    if odir:  # if odir is defined, create it if needed and move the result to it
        head, tail = os.path.split(out)
        odir = os.path.join(head, odir)
        os.makedirs(odir, exist_ok=True)
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

    cmd = CCCommand(cc_exe, silent=silent, fmt=fmt)
    cmd.open_file(cloudfile, global_shift=global_shift)
    cmd.append('-APPLY_TRANS')
    cmd.append(transformation)
    cmd.extend(['-SAVE_CLOUDS', 'file', out])
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


##########
# C2C_DIST
##########


def c2c_dist(compared, reference,
             max_dist=None, split_xyz=False, split_xy_z=False, octree_level=10,
             export_fmt='SBF', global_shift='AUTO', silent=True,
             odir=None, verbose=False, cc_exe=cc_exe):
    """
    Compute the distance between a compared cloud and a reference cloud

    :param compared: filename of the compared cloud
    :param reference: filename of the reference cloud
    :param max_dist: max distance (speed calculations)
    :param split_xyz:
    :param split_xy_z:
    :param octree_level: well set, it can speed up the calculations
    :param export_fmt: output format
    :param global_shift:
    :param silent: show the CloudCompare console
    :param odir:
    :param verbose: verbose mode
    :param cc_exe: CloudCompare executable (defaults to standard location)
    :return:
    """

    cmd = CCCommand(cc_exe, silent=silent, fmt=export_fmt)  # create the command
    cmd.open_file(compared, global_shift=global_shift)  # open compared
    cmd.open_file(reference, global_shift=global_shift)  # open reference

    cmd.append('-C2C_DIST')

    if max_dist:
        cmd.extend(['-MAX_DIST', str(max_dist)])
    cmd.extend(['-OCTREE_LEVEL', str(octree_level)])
    if split_xyz == 'split_xyz':
        cmd.append('-SPLIT_XYZ')
    elif split_xy_z == 'split_xy_z':
        cmd.append('-SPLIT_XY_Z')
    cmd.append('-POP_CLOUDS')  # remove reference cloud from the database

    # output directory
    head, tail, root, ext = misc.head_tail_root_ext(compared)
    if odir is None:
        output_directory = head
    else:
        output_directory = os.path.join(head, odir)
        if not os.path.exists(odir):
            print(f'[{__name__}] create output directory {output_directory}')
            os.makedirs(output_directory, exist_ok=True)

    if max_dist:
        out = os.path.join(output_directory, root + f'_C2C_DIST_{max_dist}m.{export_fmt.lower()}')
    else:
        out = os.path.join(output_directory, root + f'_C2C_DIST.{export_fmt.lower()}')

    cmd.extend(['-SAVE_CLOUDS', 'FILE',  out])  # save cloud

    misc.run(cmd, verbose=verbose)
    
    return out


def closest_point_set(compared, reference, silent=True, debug=False):
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


def octree_normals(cloud, radius, with_grids=False, angle=1,
                   orient='PLUS_Z', model='QUADRIC', fmt='BIN',
                   silent=True, verbose=False, global_shift='AUTO', cc=cc_exe):
    """

    Parameters
    ----------
    cloud
    radius
    orient : str
        PLUS_ZERO PLUS_ORIGIN MINUS_ZERO MINUS_ORIGIN PLUS_BARYCENTER MINUS_BARYCENTER
        PLUS_X MINUS_X
        PLUS_Y MINUS_Y
        PLUS_Z MINUS_Z
        PREVIOUS
        SENSOR_ORIGIN
        WITH_GRIDS
        WITH_SENSOR
    model : str
        LS TRI QUADRIC
    fmt
    silent
    verbose
    global_shift
    cc

    Returns
    -------

    """

    cmd = CCCommand(cc, silent=silent, fmt=fmt)  # create the command
    cmd.open_file(cloud, global_shift=global_shift)  # open compared

    cmd.extend(['-OCTREE_NORMALS', str(radius)])
    if with_grids:
        cmd.extend(['-WITH_GRIDS', str(angle)])
    cmd.extend(['-MODEL', model, '-ORIENT', orient])

    if fmt.lower() == 'bin':
        out = os.path.splitext(cloud)[0] + '_WITH_NORMALS.bin'
    else:
        raise ValueError(f'format {fmt} not supported yet? (only bin is supported)')
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd, verbose=verbose)

    return out
