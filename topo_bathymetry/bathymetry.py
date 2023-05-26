import logging
import os
import shutil

import numpy as np

from ..config.config import cc_custom, cc_std
from ..tools import cc, las, misc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# INITIAL SCALAR FIELDS
# 0 [Classif] Value (added by LAStools)
# 1 Intensity
# 2 GpsTime
# 3 ReturnNumber
# 4 NumberOfReturn
# 5 ScanDirectionFlag
# 6 EdgeOfFlightLine
# 7 ScanAngleRank
# 8 PointSourceId

i_point_source_id = 8

# 9 C2C3_XY
# 10 C2C3
# 11 C2C3_X
# 12 C2C3_Y
# 13 C2C3_Z
i_c2c3_xy = 9
i_c2c3 = 10
i_c2c3_x = 11  # REMOVED
i_c2c3_y = 12  # REMOVED
i_c2c3_z = 11  # AFTER REMOVAL OF X AND Y

# 14 Dip (degrees)
# 15 Dip direction (degrees)
# 16 Number of neighbors (r=5)
i_dip = 12
i_nn = 14

# 12 C2C absolute distances (XY)
# 13 C2C absolute distances
# 14 C2C absolute distances (X)
# 15 C2C absolute distances (Y)
# 16 C2C absolute distances (Z)

i_c2c_xy = 12
i_c2c = 13
i_c2c_x = 14
i_c2c_y = 15
i_c2c_z = 16


def get_shift(config):
    # the shift comes from
    # 1) the intensity correction, i.e. imax_minus_i and intensity_class SF have been added
    # 2) the classification field added by LAStools
    if config == 'classified':
        shift = 0
    elif config == 'i_corr_classified':
        shift = 2
    elif config == 'i_corr_not_classified':
        shift = 1
    elif config == 'not_classified':
        shift = -1
    else:
        print(f'[get_shift] config unknown: {config}')
        shift = None
    return shift


def extract_seed_from_water_surface(c3_cloud_with_c2c3_dist, water_surface, c2c3_xy_index, deepness=-0.2):
    # c3_cloud_with_c2c3_dist shall contain C2C3_Z, C2C3 and C2C3_XY scalar fields
    head, tail, root, ext = misc.head_tail_root_ext(c3_cloud_with_c2c3_dist)
    odir = os.path.join(head, 'bathymetry')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + f'_bathymetry_seed.bin')

    cmd = cc_custom
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT BIN -AUTO_SAVE OFF'
    cmd += f' -O {c3_cloud_with_c2c3_dist}'  # global shift not needed with a bin file
    cmd += f' -O {water_surface}'  # global shift not needed with a bin file
    cmd += ' -C2C_DIST -SPLIT_XY_Z'
    cmd += ' -POP_CLOUDS'
    xy_index = c2c3_xy_index + 3
    cmd += f' -SET_ACTIVE_SF {xy_index} -FILTER_SF 0 5.'  # C2C XY
    cmd += f' -SET_ACTIVE_SF {xy_index + 4} -FILTER_SF MIN {deepness}'  # C2C Z
    # prevent the collection of points being above the water surface
    cmd += f' -SET_ACTIVE_SF {c2c3_xy_index + 2} -FILTER_SF MIN {deepness}'  # C2C3 Z
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd, verbose=True, advanced=True)

    return out


def propagate(c3_cloud_with_c2c3_dist, current_bathymetry, c2c3_xy_index, deepness=-0.2, step=None):
    # c3_cloud_with_c2c3_dist shall contain C2C3_Z, C2C3 and C2C3_XY scalar fields
    head, tail, root, ext = misc.head_tail_root_ext(c3_cloud_with_c2c3_dist)
    odir = os.path.join(head, 'bathymetry')
    if step is not None:
        out = os.path.join(odir, root + f'_propagation_step_{step}.bin')
    else:
        out = os.path.join(odir, root + f'_propagation.bin')
    dip = np.tan(1. * np.pi / 180)  # dip 1 degree

    cmd = cc_custom
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT BIN -AUTO_SAVE OFF'
    cmd += f' -O {c3_cloud_with_c2c3_dist}'
    cmd += f' -O {current_bathymetry}'
    cmd += ' -C2C_DIST -SPLIT_XY_Z'
    cmd += ' -POP_CLOUDS'
    xy_index = c2c3_xy_index + 3
    cmd += f' -SET_ACTIVE_SF {xy_index} -FILTER_SF 0.001 10.'  # keep closest points, no duplicates (i.e. xy = 0)
    cmd += f' -SET_ACTIVE_SF {xy_index + 4} -FILTER_SF -0.1 0.1'
    cmd += f' -SET_ACTIVE_SF {c2c3_xy_index + 2} -FILTER_SF MIN {deepness}'  # consider only points with C3 below C2
    cmd += f' -O {current_bathymetry} -MERGE_CLOUDS'  # merge new points with the previous ones
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd, verbose=True, advanced=True)

    return out


def c2c_class_16(line, class_16, global_shift, octree_level=10, shift=2):
    print(f'processing line {line}')

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, 'c2c_class_16')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')

    x, y, z = global_shift

    print("[c2c_class_16]")
    cmd = cc_custom
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {line}'  # compared
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {class_16}'  # reference
    cmd += f' -C2C_DIST -MAX_DIST 10 -OCTREE_LEVEL {octree_level} -SPLIT_XY_Z'
    cmd += ' -POP_CLOUDS'  # remove class_16 from the database
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd, verbose=True)

    return out


def get_class_16_hd(c2c, zmin_zmax=(-0.2, 0.2), depth_min=0, xy_max=2, lastools_gc=True):
    print(f'process {c2c}')
    head, tail, root, ext = misc.head_tail_root_ext(c2c)
    odir = os.path.join(head, 'with_16')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')

    pc, sf, config = cc.read_sbf(c2c)
    name_index = cc.get_name_index_dict(config)
    zmin, zmax = zmin_zmax

    xy = sf[:, name_index['C2C absolute distances[<10] (XY)']]
    z = sf[:, name_index['C2C absolute distances[<10] (Z)']]
    depth = sf[:, name_index['depth']]
    if lastools_gc:
        classification = sf[:, name_index['Classification']]
    else:
        classification = np.zeros(z.shape)

    select_16 = (depth < depth_min) & (xy < xy_max) & (zmin < z) & (z < zmax)
    classification[select_16] = 16
    print(f'{np.count_nonzero(select_16)}/{len(classification)} classified as bathymetry (class 16)')

    # remove unused scalar fields
    sf, config = cc.remove_sf('C2C absolute distances[<10] (XY)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<10]', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<10] (X)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<10] (Y)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<10] (Z)', sf, config)

    cc.write_sbf(out, pc, sf, config)

    return out


def c2c_class_15(line, class_15, global_shift, octree_level=10, shift=2):
    print(f'processing line {line}')

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, 'c2c_class_15')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')
    if os.path.exists(out):
        print(f'nothing to do, out already exists: {out}')
        return out

    x, y, z = global_shift

    print("[c2c_class_15]")
    cmd = cc_custom
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {line}'  # compared
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {class_15}'  # reference
    cmd += f' -C2C_DIST -MAX_DIST 5 -OCTREE_LEVEL {octree_level} -SPLIT_XY_Z'
    cmd += ' -POP_CLOUDS'  # remove class_15 from the database
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd, advanced=True, verbose=True)

    return out


def get_class_15_hd(line, zmin_zmax=(-0.5, 1), xy_max=1, depth_min=0, lastools_gc=True):

    print(f'processing line {line}')

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, 'with_15')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')

    pc, sf, config = cc.read_sbf(line)
    zmin, zmax = zmin_zmax

    name_index = cc.get_name_index_dict(config)

    # the field "Classification" should be present because class 16 classification has already been done
    classification = sf[:, name_index['Classification']]
    depth = sf[:, name_index['depth']]
    xy = sf[:, name_index['C2C absolute distances[<5] (XY)']]
    z = sf[:, name_index['C2C absolute distances[<5] (Z)']]

    select_15 = (depth < depth_min) & (xy < xy_max) & (zmin < z) & (z < zmax)
    classification[select_15] = 15
    print(f'{np.count_nonzero(select_15)}/{len(classification)} classified as water volume (class 15)')

    sf, config = cc.remove_sf('C2C absolute distances[<5] (XY)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<5]', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<5] (X)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<5] (Y)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<5] (Z)', sf, config)

    cc.write_sbf(out, pc, sf, config)

    return out


def replace_class_in_line(line, class_, lines_15_16_dir, global_shift, in_place=False):

    head, tail, root, ext = misc.head_tail_root_ext(line)
    if in_place:
        out = line
        print('WARNING merge will be done in place')
    else:
        odir = os.path.join(head, 'lines_1_2_5_6_15_16')
        os.makedirs(odir, exist_ok=True)
        out = os.path.join(odir, tail)
    # the name of the class_hd is very similar to the name of the line
    class_hd = os.path.join(lines_15_16_dir, root + f'_{class_}.laz')

    if os.path.exists(class_hd):
        print(f'   REPLACE {class_hd}\n   IN {line}')
    else:
        print(f'nothing to merge for line {line} (class {class_})')
        return None

    x, y, z = global_shift

    i__ = 10

    print(f"[replace_class_in_line] replace points in the line with class {class_} points")
    cmd = cc_std
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT LAS -EXT laz -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {line}'  # compared
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {class_hd}'  # reference
    cmd += ' -C2C_DIST -MAX_DIST 10 -OCTREE_LEVEL 10'
    cmd += ' -POP_CLOUDS'  # remove class_hd from the database before filtering
    cmd += f' -SET_ACTIVE_SF {i__} -FILTER_SF 0.05 MAX'  # remove duplicates (5cm)
    cmd += f' -REMOVE_SF {i__}'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {class_hd}'  # re-open class_hd for merging
    cmd += f' -MERGE_CLOUDS'
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd, verbose=True)

    return out


def get_fwf_from_class_15(line, class_15, n_scalar_fields, global_shift=None, octree_level=11, silent=True):
    # c2_cloud_with_c2c3_dist shall contain C2C3_Z, C2C3 and C2C3_XY scalar fields
    if not misc.exists(line):
        return
    if not misc.exists(class_15):
        return

    print(f'[get_fwf_from_class_15] processing {line}')

    head_15, tail_15, root_15, ext_15 = misc.head_tail_root_ext(class_15)
    head_line, tail_line, root_line, ext_line = misc.head_tail_root_ext(line)
    odir = os.path.join(head_line, 'selection')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root_line + '.sbf')

    cmd = cc_std
    if silent:
        cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    else:
        cmd += ' -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'

    if global_shift:
        x, y, z = global_shift
        cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {line}'
        cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {class_15}'
    else:
        cmd += f' -O {line}'
        cmd += f' -O {class_15}'
    if octree_level is not None:
        cmd += f' -C2C_DIST -SPLIT_XYZ -MAX_DIST 20 -OCTREE_LEVEL {octree_level}'  # 1st = compared / 2nd = reference
    else:
        cmd += f' -C2C_DIST -SPLIT_XYZ -MAX_DIST 20'  # 1st = compared / 2nd = reference
    cmd += ' -POP_CLOUDS'
    # keep points around class 15, order of the scalar fields Z / _ / X / Y
    i_z = n_scalar_fields
    i_ = n_scalar_fields + 1
    cmd += f' -SET_ACTIVE_SF {i_z} -FILTER_SF -10 2'  # filter wrt Z
    cmd += f' -SET_ACTIVE_SF {i_} -FILTER_SF MIN 10'  # filter wrt to absolute distances
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd)

    return out


def extract_lines_from_class_16_fwf(class_16_fwf, id_name):
    head, tail = os.path.split(class_16_fwf)
    data = las.read(class_16_fwf, extra_field=True)
    ids = np.unique(data.point_source_id)

    n = data['point_source_id'].size
    for point_source_id in ids:
        selection = (data.point_source_id == point_source_id)
        print(f'point_source_id {point_source_id}: keep {np.count_nonzero(selection)}/{n} available points')
        out = os.path.join(head, id_name[point_source_id])
        out_data = las.filter_las(data, selection)
        extra = [(("depth", "float32"), out_data["depth"])]
        las.WriteLAS(out, out_data, extra_fields=extra)
        print(f'write {out}')


def merge_discrete_and_fwf(lines, dir_16_fwf, in_place=False):
    no_merge = []
    for line in lines:
        head, tail = os.path.split(line)
        odir = os.path.join(head, 'discrete_and_fwf_merged')
        os.makedirs(odir, exist_ok=True)
        line_16_fwf = os.path.join(dir_16_fwf, tail)
        if not os.path.exists(line_16_fwf):
            print('nothing to merge, copy line in output directory')
            no_merge.append(line)
            out = shutil.copy(line, odir)
        else:
            cmd = cc_std
            if in_place:
                out = line
            else:
                out = os.path.join(odir, tail)
            cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT LAS -EXT laz -AUTO_SAVE OFF'
            cmd += f' -O -GLOBAL_SHIFT AUTO {line}'
            cmd += f' -O -GLOBAL_SHIFT FIRST {line_16_fwf}'
            cmd += ' -MERGE_CLOUDS'
            cmd += f' -SAVE_CLOUDS FILE {out}'
            misc.run(cmd)

        print(f'output file: {out}')

    return no_merge


def add_depth(line, water_surface, global_shift, octree_level=10, remove_extra_sf=False, silent=True):
    print(f'processing line {line}')

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, 'with_depth')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')

    x, y, z = global_shift

    print("[add_depth]")
    # be careful of global_shift, bug corrected in CloudCompare but maybe not merged in the last release
    # use modified version of CloudCompare
    # use the same version of CloudCompare afterwards to avoid incompatibilities with SBF?
    # cmd = cc_2022_07_05
    cmd = cc_std
    if silent:
        cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    else:
        cmd += ' -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {line}'  # compared
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {water_surface}'  # reference
    cmd += f' -C2C_DIST -OCTREE_LEVEL {octree_level} -MAX_DIST 350 -SPLIT_XYZ'
    cmd += ' -POP_CLOUDS'  # remove water_surface from the database
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd)

    # open sbf and remove unused scalar fields (the remove_sf option is not working in CloudCompare command line)

    pc, sf, config = cc.read_sbf(out)

    cc.rename_sf('C2C absolute distances[<350] (Z)', 'depth', config)

    # remove unused scalar fields
    sf, config = cc.remove_sf('C2C absolute distances[<350]', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350] (X)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350] (Y)', sf, config)
    if remove_extra_sf:
        sf, config = cc.remove_sf('intensity_class', sf, config)
        sf, config = cc.remove_sf('imax_minus_i', sf, config)
        sf, config = cc.remove_sf('UserData', sf, config)

    cc.write_sbf(out, pc, sf, config)

    return out


def add_depth(line, water_surface, global_shift, octree_level=10, remove_extra_sf=False, silent=True, cc_exe=cc_std):
    print(f'processing line {line}')

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, 'with_depth')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')

    x, y, z = global_shift

    print("[add_depth]")
    # be careful of global_shift, bug corrected in CloudCompare but maybe not merged in the last release
    # use modified version of CloudCompare
    # use the same version of CloudCompare afterwards to avoid incompatibilities with SBF?
    # cmd = cc_2022_07_05
    cmd = cc_exe
    if silent:
        cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    else:
        cmd += ' -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {line}'  # compared
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {water_surface}'  # reference
    cmd += f' -C2C_DIST -OCTREE_LEVEL {octree_level} -MAX_DIST 350 -SPLIT_XYZ'
    cmd += ' -POP_CLOUDS'  # remove water_surface from the database
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd)

    # open sbf and remove unused scalar fields (the remove_sf option is not working in CloudCompare command line)

    pc, sf, config = cc.read_sbf(out)

    cc.rename_sf('C2C absolute distances[<350] (Z)', 'depth', config)

    # remove unused scalar fields
    sf, config = cc.remove_sf('C2C absolute distances[<350]', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350] (X)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350] (Y)', sf, config)
    if remove_extra_sf:
        sf, config = cc.remove_sf('intensity_class', sf, config)
        sf, config = cc.remove_sf('imax_minus_i', sf, config)
        sf, config = cc.remove_sf('UserData', sf, config)

    cc.write_sbf(out, pc, sf, config)

    return out


def add_depth_laz(line, water_surface, global_shift, octree_level=10, silent=True, cc_exe=cc_std):
    print(f'processing line {line}')

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, 'with_depth')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.laz')

    x, y, z = global_shift

    print("[add_depth]")

    cmd = [cc_exe]
    if silent:
        cmd.append('-SILENT')
    cmd.append('-NO_TIMESTAMP')
    cmd.append('-C_EXPORT_FMT')
    cmd.extend(['LAS', '-EXT', 'LAZ'])
    cmd.extend(['-AUTO_SAVE', 'OFF'])

    cmd.extend(['-O', '-GLOBAL_SHIFT', str(x), str(y), str(z), line])  # open data
    cmd.extend(['-O', '-GLOBAL_SHIFT', str(x), str(y), str(z), water_surface])  # open water surface

    # compute cloud to cloud distances
    cmd += ['-C2C_DIST', '-OCTREE_LEVEL', str(octree_level), '-MAX_DIST', '350', '-SPLIT_XYZ']

    cmd.append('-POP_CLOUDS')  # remove water_surface from the database

    # remove unused scalar fields
    cmd += ['-REMOVE_SF', "'C2C absolute distances[<350]'"]
    cmd += ['-REMOVE_SF', "'C2C absolute distances[<350] (X)'"]
    cmd += ['-REMOVE_SF', "'C2C absolute distances[<350] (Y)'"]

    cmd += ['-RENAME_SF', 'LAST', 'depth']  # rename the scalar field 'C2C absolute distances[<350] (Z)' to 'depth'

    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd)

    return out
