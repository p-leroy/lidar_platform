import os
import shutil

import numpy as np

from ..tools import cc, misc
from ..config.config import cc_std, cc_custom
from . import bathymetry as bathy


def c2c_c2c3(compared, reference, xy_index, global_shift):
    # compute cloud to cloud distances and rename the scalar fields for further processing
    head, tail = os.path.split(compared)
    root, ext = os.path.splitext(tail)
    out = os.path.join(head, root + '_C2C3.bin')

    cmd = cc_custom
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT BIN -AUTO_SAVE OFF'
    # if global_shift == 'auto':
    #     cmd += f' -O -GLOBAL_SHIFT AUTO {compared}'
    # else:
    #     shift_x = global_shift[0]
    #     shift_y = global_shift[1]
    #     shift_z = global_shift[2]

    x, y, z = global_shift
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {compared}'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {reference}'
    cmd += f' -C2C_DIST -SPLIT_XY_Z'
    cmd += ' -POP_CLOUDS'
    # XY _ X Y Z scalar fields are in this order
    cmd += f' -REMOVE_SF {xy_index + 3}'  # remove Y first
    cmd += f' -REMOVE_SF {xy_index + 2}'   # then remove X
    cmd += f' -RENAME_SF {xy_index + 2} C2C3_Z'
    cmd += f' -RENAME_SF {xy_index + 1} C2C3'
    cmd += f' -RENAME_SF {xy_index} C2C3_XY'
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd)

    return out


def extract_seed(cloud, c2c3_xy_index, deepness=0.5):
    if not misc.exists(cloud):
        return
    head, tail = os.path.split(cloud)
    head, tail, root, ext = misc.head_tail_root_ext(cloud)
    odir = os.path.join(head, 'water_surface')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + f'_water_surface_seed.bin')

    cmd = cc_std
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT BIN -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT AUTO {cloud}'
    cmd += f' -SET_ACTIVE_SF {c2c3_xy_index + 2} -FILTER_SF {deepness} 5.'  # C2C3_Z i.e. depth
    cmd += ' -OCTREE_NORMALS 5. -MODEL LS -ORIENT PLUS_Z -NORMALS_TO_DIP'
    cmd += f' -SET_ACTIVE_SF {c2c3_xy_index + 3} -FILTER_SF MIN 1.'  # DIP
    cmd += ' -DENSITY 5. -TYPE KNN'
    cmd += f' -SET_ACTIVE_SF LAST -FILTER_SF 10 MAX'
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd)

    return out


def propagate_1deg(c2_cloud_with_c2c3_dist, current_surface, c2c3_xy_index, deepness=0.2, step=None):
    # c2_cloud_with_c2c3_dist shall contain C2C3_XY, C2C3 and C2C3_Z scalar fields
    if not misc.exists(c2_cloud_with_c2c3_dist):
        return
    if not misc.exists(current_surface):
        return
    head, tail, root, ext = misc.head_tail_root_ext(c2_cloud_with_c2c3_dist)
    odir = os.path.join(head, 'water_surface')

    if step is not None:
        out = os.path.join(odir, root + f'_propagation_{step}.bin')
    else:
        out = os.path.join(odir, root + f'_propagation.bin')
    dip = np.tan(1. * np.pi / 180)  # dip 1 degree

    cmd = cc_custom
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT BIN -AUTO_SAVE OFF'
    cmd += f' -O {c2_cloud_with_c2c3_dist}'  # no global shift for bin file
    cmd += f' -O {current_surface}'  # no global shift for a bin file
    cmd += ' -C2C_DIST -SPLIT_XY_Z'
    cmd += ' -POP_CLOUDS'
    xy_index = c2c3_xy_index + 3
    cmd += f' -SET_ACTIVE_SF {xy_index} -FILTER_SF 0.001 10.'  # keep closest points and avoid duplicates (i.e. xy = 0)
    cmd += f' -SET_ACTIVE_SF {c2c3_xy_index + 2} -FILTER_SF {deepness} MAX'  # consider only points with C2 above C3
    cmd += f' -SF_OP_SF {xy_index + 4} DIV {xy_index}'  # compute the dip: Z / XY
    cmd += f' -SET_ACTIVE_SF LAST -FILTER_SF {-dip} {dip}'  # filter wrt dip
    cmd += f' -O -GLOBAL_SHIFT FIRST {current_surface} -MERGE_CLOUDS' # merge new points with the previous ones
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd)

    return out


def c2c_class_9(line, class_9, global_shift, octree_level=10, odir='c2c_class_9'):

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, odir)
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + f'.sbf')

    if os.path.exists(out):
        print(f'[line_c2c_class_9] sbf already exists, nothing to do {out}')
        return out

    x, y, z = global_shift
    print(f'process line {line}')
    cmd = cc_custom
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {line}'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {class_9}'
    cmd += f' -C2C_DIST -SPLIT_XY_Z -MAX_DIST 350 -OCTREE_LEVEL {octree_level}'
    cmd += ' -POP_CLOUDS'  # remove class_9 from the database
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd)

    return out


def classify_class_9_in_line(c2c, zmin_zmax=(-0.2, 0.05), xy_max=5, lastools_gc=True):
    # look for c2c results
    head, tail, root, ext = misc.head_tail_root_ext(c2c)
    odir = os.path.join(head, 'lines_with_9')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')

    if os.path.exists(out):
        print(f'[classify_class_9_in_line] sbf already exists, nothing to do {out}')
        return out

    pc, sf, config = cc.read_sbf(c2c)
    zmin, zmax = zmin_zmax

    name_index = cc.get_name_index_dict(config)
    xy = sf[:, name_index['C2C absolute distances[<350] (XY)']]
    z = sf[:, name_index['C2C absolute distances[<350] (Z)']]

    if lastools_gc:
        classification = sf[:, name_index['[Classif] Value']]
        cc.rename_sf('[Classif] Value', 'Classification', config)
    else:
        classification = np.zeros(z.shape)

    # NOTE: along z, the standard deviation is important, the water surface is extracted from the lowest points,
    # so all other points belonging to the water surface will be above...
    select_9 = (xy < xy_max) & (zmin < z) & (z < zmax)
    classification[select_9] = 9
    print(f'{np.count_nonzero(select_9)}/{len(classification)} classified as water surface (class 9)')

    if not lastools_gc:
        cc.add_sf('Classification', sf, classification)

    # remove unused scalar fields
    sf, config = cc.remove_sf('UserData', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350] (XY)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350]', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350] (X)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350] (Y)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<350] (Z)', sf, config)

    cc.write_sbf(out, pc, sf, config)

    return out


def c2c_class_15_16(line, class_15_16, global_shift, octree_level=10, odir='c2c_class_15_16'):

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, odir)
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + f'.sbf')

    if os.path.exists(out):
        print(f'[line_c2c_class_9] sbf already exists, nothing to do {out}')
        return out

    x, y, z = global_shift
    print(f'process line {line}')
    cmd = cc_custom
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {line}'
    cmd += f' -O -GLOBAL_SHIFT {x} {y} {z} {class_15_16}'
    cmd += f' -C2C_DIST -SPLIT_XY_Z -MAX_DIST 20 -OCTREE_LEVEL {octree_level}'
    cmd += ' -POP_CLOUDS'  # remove class_9 from the database
    cmd += f' -SAVE_CLOUDS FILE {out}'
    misc.run(cmd)

    return out


def reclassify_class_2_using_class_15_16(c2c, xy_max=3):
    # look for c2c results
    head, tail, root, ext = misc.head_tail_root_ext(c2c)
    odir = os.path.join(head, 'lines_with_2_corr')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')

    if os.path.exists(out):
        print(f'[classify_class_9_in_line] sbf already exists, nothing to do {out}')
        return out

    pc, sf, config = cc.read_sbf(c2c)

    name_index = cc.get_name_index_dict(config)
    xy = sf[:, name_index['C2C absolute distances[<20] (XY)']]

    try:
        classification = sf[:, name_index['Classification']]
    except KeyError:
        try:
            classification = sf[:, name_index['[Classif] Value']]
        except KeyError:
            raise

    # reclassify ground points in C2 which are too close to class 15 and 16 of C3
    # this is to avoid having water surface points classified as ground points (where we have C3 data)
    select = (classification == 2) & (xy < xy_max)
    classification[select] = 1
    print(f'{np.count_nonzero(select)}/{len(classification)} reclassified as undetermined (1 instead of 2)')

    # remove unused scalar fields
    sf, config = cc.remove_sf('C2C absolute distances[<20] (XY)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<20]', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<20] (X)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<20] (Y)', sf, config)
    sf, config = cc.remove_sf('C2C absolute distances[<20] (Z)', sf, config)

    cc.write_sbf(out, pc, sf, config)

    return out


def reclassify_class_2_using_intensity(sbf, i_min):
    head, tail = os.path.split(sbf)
    odir = os.path.join(head, 'i_selection')
    os.makedirs(odir, exist_ok=True)

    pc, sf, config = cc.read_sbf(sbf)
    name_index = cc.get_name_index_dict(config)
    classification = sf[:, name_index['Classification']]
    intensity = sf[:, name_index['Intensity']]

    select = (classification == 2) & (intensity < i_min)

    classification[select] = 1  # reclass points as undetermined
    print(f'{np.count_nonzero(select)}/{len(classification)} reclassified as undetermined (1 instead of 2)')

    out = os.path.join(odir, tail)
    cc.write_sbf(out, pc, sf, config)

    return out
