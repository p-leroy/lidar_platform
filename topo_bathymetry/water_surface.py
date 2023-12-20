import os

import numpy as np

from ..tools import cc, misc, sbf
from ..config.config import cc_exe


def c2c_c2c3(compared, reference, global_shift):
    # compute cloud to cloud distances and rename the scalar fields for further processing
    head, tail = os.path.split(compared)
    root, ext = os.path.splitext(tail)
    out = os.path.join(head, root + '_C2C3.bin')

    cmd = [cc_exe]
    cmd.extend(['-SILENT', '-NO_TIMESTAMP', '-C_EXPORT_FMT', 'BIN', '-AUTO_SAVE', 'OFF'])

    x, y, z = global_shift
    cmd.extend(['-O', '-GLOBAL_SHIFT', str(x), str(y), str(z), compared])
    cmd.extend(['-O', '-GLOBAL_SHIFT', str(x), str(y), str(z), reference])
    cmd.extend(['-C2C_DIST', '-SPLIT_XY_Z'])
    cmd.append('-POP_CLOUDS')
    # XY _ X Y Z scalar fields are in this order
    cmd.extend(['-REMOVE_SF', 'C2C absolute distances (X)'])
    cmd.extend(['-REMOVE_SF', 'C2C absolute distances (Y)'])
    cmd.extend(['-RENAME_SF', 'C2C absolute distances (Z)', 'C2C3_Z'])
    cmd.extend(['-RENAME_SF', 'C2C absolute distances', 'C2C3'])
    cmd.extend(['-RENAME_SF', 'C2C absolute distances (XY)', 'C2C3_XY'])
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])
    misc.run(cmd)

    return out


def extract_seed(cloud, depth=0.5):
    if not misc.exists(cloud):
        return
    head, tail = os.path.split(cloud)
    head, tail, root, ext = misc.head_tail_root_ext(cloud)
    odir = os.path.join(head, 'water_surface')
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + f'_water_surface_seed.bin')

    cmd = [cc_exe]
    cmd.extend(['-SILENT', '-NO_TIMESTAMP', '-C_EXPORT_FMT', 'BIN', '-AUTO_SAVE', 'OFF'])
    cmd.extend(['-O', '-GLOBAL_SHIFT', 'AUTO', cloud])
    cmd.extend(['-SET_ACTIVE_SF', f'C2C3_Z', '-FILTER_SF', str(depth), str(5.)])  # C2C3_Z i.e. depth
    cmd.extend(['-OCTREE_NORMALS', str(5.), '-MODEL', 'LS', '-ORIENT', 'PLUS_Z', '-NORMALS_TO_DIP'])
    cmd.extend(['-SET_ACTIVE_SF', 'Dip (degrees)', '-FILTER_SF', 'MIN', str(1.)])  # DIP
    cmd.extend(['-DENSITY', str(5.), '-TYPE', 'KNN'])
    cmd.extend(['-SET_ACTIVE_SF', 'LAST', '-FILTER_SF', str(10), 'MAX'])
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])
    misc.run(cmd)

    return out


def propagate_1deg(c2_cloud_with_c2c3_dist, current_surface, depth=0.2, step=None):
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

    cmd = cc.CCCommand(cc_exe, silent=True, fmt='BIN')
    cmd.open_file(c2_cloud_with_c2c3_dist)  # no global shift for bin file
    cmd.open_file(current_surface)  # no global shift for bin file
    cmd.extend(['-C2C_DIST', '-SPLIT_XY_Z'])
    cmd.append('-POP_CLOUDS')

    # keep closest points and avoid duplicates (i.e. xy = 0)
    cmd.extend(['-SET_ACTIVE_SF', 'C2C absolute distances (XY)', '-FILTER_SF', str(0.001), str(10.)])
    # consider only points with C2 above C3
    cmd.extend(['-SET_ACTIVE_SF', 'C2C3_Z', '-FILTER_SF', str(depth), 'MAX'])
    # compute the dip: Z / XY
    cmd.extend(['-SF_OP_SF', 'C2C absolute distances (Z)', 'DIV', 'C2C absolute distances (XY)'])
    # filter wrt dip
    cmd.extend(['-SET_ACTIVE_SF', 'Dip (degrees)', '-FILTER_SF', str(-dip), str(dip)])
    # merge new points with the previous ones
    cmd.open_file(current_surface, global_shift='FIRST')
    cmd.append('-MERGE_CLOUDS')
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])
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
    cmd = cc_exe
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

    sbf_data = sbf.read(c2c)
    pc, sf, config = sbf_data.pc, sbf_data.sf, sbf_data.config
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


def c2c_class_15_16(line, class_15_16, global_shift, octree_level=10, odir='c2c_15_16'):

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, odir)
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + f'.sbf')

    if os.path.exists(out):
        print(f'[line_c2c_class_9] sbf already exists, nothing to do {out}')
        return out

    x, y, z = global_shift
    print(f'process line {line}')
    cmd = [cc_exe]
    cmd.extend(['-SILENT', '-NO_TIMESTAMP', '-C_EXPORT_FMT', 'SBF', '-AUTO_SAVE', 'OFF'])
    cmd.extend(['-O', '-GLOBAL_SHIFT', str(x), str(y), str(z), line])
    cmd.extend(['-O', '-GLOBAL_SHIFT', str(x), str(y), str(z), class_15_16])
    cmd.extend(['-C2C_DIST', '-SPLIT_XY_Z', '-MAX_DIST', str(10), '-OCTREE_LEVEL', str(octree_level)])
    cmd.append('-POP_CLOUDS')  # remove class_9 from the database
    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])
    misc.run(cmd)

    return out


def reclassify_class_2_using_class_15_16(c2c, xy_max=3, dir_name='with_2_corr'):
    # look for c2c results
    if not os.path.exists(c2c):
        print('WARNING reclassify_class_2_using_class_15_16 failed silently because c2c file does not exists')
        return
    head, tail, root, ext = misc.head_tail_root_ext(c2c)
    odir = os.path.join(head, dir_name)
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, root + '.sbf')

    if os.path.exists(out):
        print(f'[classify_class_9_in_line] sbf already exists, nothing to do {out}')
        return out

    sbf_data = sbf.read(c2c)
    sf = sbf_data.sf

    name_index = sbf_data.get_name_index_dict()
    dist = sf[:, name_index['C2C absolute distances[<10]']]
    dist_xy = sf[:, name_index['C2C absolute distances[<10] (XY)']]

    try:
        classification = sf[:, name_index['Classification']]
    except KeyError:
        try:
            classification = sf[:, name_index['[Classif] Value']]
        except KeyError:
            raise

    # reclassify ground points in C2 which are too close to class 15 and 16 of C3
    # this is to avoid having water surface points classified as ground points (where we have C3 data)
    select = (classification == 2) & (dist_xy < xy_max) & (dist < 10)
    classification[select] = 1
    print(f'{np.count_nonzero(select)}/{len(classification)} reclassified as undetermined (1 instead of 2)')

    # remove unused scalar fields
    sbf_data.remove_sf('C2C absolute distances[<10] (XY)')
    sbf_data.remove_sf('C2C absolute distances[<10]')
    sbf_data.remove_sf('C2C absolute distances[<10] (X)')
    sbf_data.remove_sf('C2C absolute distances[<10] (Y)')
    sbf_data.remove_sf('C2C absolute distances[<10] (Z)')

    sbf.write(out, sbf_data.pc, sbf_data.sf, sbf_data.config)

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
