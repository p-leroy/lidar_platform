# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:56:40 2021

@author: Paul Leroy
"""

import os

from lidar_platform import misc

#bin_ = 'C:/opt/LAStools/bin'
bin_ = 'G:/LIDAR_softwares_manuals/LAStools/bin'


def exe(cmd, args, debug=False):
    cmd = os.path.join(bin_, cmd)
    for switch in args:
        if args[switch] is not None:
            cmd += f' -{switch} {args[switch]}'
        else:
            cmd += f' -{switch}'
    print(cmd)
    return misc.run(cmd, debug=debug)


def las2las(fullname, utm='59south', target_epsg='2193', debug=False):
    cmd = os.path.join(bin_, 'las2las')
    out = os.path.splitext(fullname)[0] + f'_{target_epsg}.laz'
    args = f' -i {fullname} -o {out}'
    args += f' -utm {utm} -meter -elevation_meter'
    args += f' -target_epsg {target_epsg}'
    print(cmd + args)
    misc.run(cmd + args, debug=debug)
    return out


def lasboundary(fullname, odir='boundaries', debug=False):
    cmd = os.path.join(bin_, 'lasboundary')
    head, tail = os.path.split(fullname)
    odir = os.path.join(head, odir)
    os.makedirs(odir, exist_ok=True)
    args = f' -i {fullname} -odir {odir} -oshp'
    print(cmd + args)
    misc.run(cmd + args, debug=debug)


def lasgrid(fullname, step, odir='grid', debug=False, method='lowest', fmt='laz'):
    cmd = [os.path.join(bin_, 'lasgrid')]
    head, tail = os.path.split(fullname)
    root, ext = os.path.splitext(fullname)
    if debug is True:
        cmd.append('-v')
    cmd.append('-i')
    cmd.append(fullname)
    odir = os.path.join(head, odir)
    os.makedirs(odir, exist_ok=True)
    cmd.append('-odir')
    cmd.append(odir)
    if '*' in tail:
        cmd.append(f'-o{fmt}')
    else:
        out = root + f'_grid({step}){method}.{fmt}'
        cmd.append('-o')
        cmd.append(out)
    cmd.append('-step')
    cmd.append(str(step))
    cmd.append(f'-{method}')

    if False:
        cmd = os.path.join(bin_, 'lasgrid')
        args = f' -i {fullname} -odir {odir} -o {out} -step {step} -{method}'
        print(cmd + args)
        misc.run(cmd + args, debug=debug)
        return out
    else:
        print(cmd)
        misc.run(cmd, debug=debug)
        return odir


def lasground(idir, i, odir, fine=None, debug=False):
    cmd = os.path.join(bin_, 'lasground')
    fullname = os.path.join(idir, i, '*.laz')
    head = os.path.join(idir, i)
    out = os.path.join(head, odir)
    try:
        os.makedirs(out, exist_ok=True) 
    except OSError:
        print("Creation of the directory %s failed" % out) 
    else: 
        print("Output directory: %s " % out)
    if debug is True:
        cmd += ' -v'
    args = ''
    if fine is not None:
        # For very steep hills you can intensify the search for initial ground 
        # points with '-fine' or '-extra_fine'
        args += f' -{fine}'
    args += f' -i {fullname} -odir {out} -olaz -cores 4'
    print(cmd + args)
    return misc.run(cmd + args, debug=debug)


def lasindex(i, debug=False):
    cmd = os.path.join(bin_, 'lasindex')
    args = f' -i {i}'
    print(cmd + args)
    return misc.run(cmd + args, debug=debug)


def lasinfo(idir, i, debug=False):
    cmd = os.path.join(bin_, 'lasinfo')
    fullname = os.path.join(idir, i)
    if debug is True:
        cmd += ' -v'
    args = f' -i {fullname} -odix _info -otxt'
    print(cmd + args)
    return misc.run(cmd + args, debug=debug)


def lasmerge(idir, i, odir, o, debug=False):
    cmd = os.path.join(bin_, 'lasmerge')
    fullname = os.path.join(idir, *i, '*.laz')
    if debug is True:
        cmd += ' -v'
    args = f' -i {fullname} -odir {odir} -o {o}'
    print(cmd + args)
    ret = misc.run(cmd + args, debug=debug)
    return os.path.join(odir, o)


def lasnoise(i, odir, step=None, isolated=None, cores=None, verbose=True):
    cmd = os.path.join(bin_, 'lasnoise')
    if verbose is True:
        cmd += ' -v'
    args = f' -i {i} -remove_noise'
    if step is not None:
        args += f' -step {step}'
    if isolated is not None:
        args += f' -isolated {isolated}'
    args += f' -odir {odir} -odix _denoised -olaz'
    if cores is not None:
        args += f' -cores {cores}'
    print(cmd + args)
    ret = misc.run(cmd + args)
    return os.path.join(odir)


def lassplit(fullname, odir='split', method=None, keep=None, debug=False):
    cmd = os.path.join(bin_, 'lassplit')
    head, tail = os.path.split(fullname)
    if isinstance(odir, list):
        odir = os.path.join(head, *odir)
    else:
        odir = os.path.join(head, odir)
    os.makedirs(odir, exist_ok=True)
    if debug is True:
        cmd += ' -v'
    # available methods: by_classification, keep_class
    if method is not None:
        if keep is not None:
            args = f' -i {fullname} -{method} -keep_class {keep} -odir {odir} -olaz'
        else:
            args = f' -i {fullname} -{method}  -odir {odir} -olaz'
    else:
        args = f' -i {fullname}  -odir {odir} -olaz'
    print(cmd + args)
    return misc.run(cmd + args, debug=debug)


def lastile(fullname, odir, tile_size=1000, buffer=20, debug=False):
    cmd = os.path.join(bin_, 'lastile')
    head, tail = os.path.split(fullname)
    out = os.path.join(head, odir)
    try:
        os.makedirs(out, exist_ok=True) 
    except OSError:
        print("Creation of the directory %s failed" % out) 
    else: 
        print("Output directory: %s " % out)
    if debug is True:
        cmd += ' -v'
    args = f' -i {fullname} -set_classification 0 -set_user_data 0'
    args += f' -tile_size {tile_size} -buffer {buffer}'
    args += f' -odir {out} -olaz'
    print(cmd + args)
    return misc.run(cmd + args, debug=debug)


def remove_buffer(idir, i, debug=False):
    cmd = os.path.join(bin_, 'lastile')
    fullname = os.path.join(idir, *i, '*.laz')
    out = os.path.join(idir, *i, 'no_buffer')
    try:
        os.makedirs(out, exist_ok=True) 
    except OSError:
        print("Creation of the directory %s failed" % out) 
    else: 
        print("Output directory: %s " % out)
    if debug is True:
        cmd += ' -v'
    args = f' -i {fullname} -remove_buffer -odir {out} -olaz -cores 4'
    print(cmd + args)
    return misc.run(cmd + args, debug=debug)


def build_gnd(fullname):
    head, tail = os.path.split(fullname)
    # 1. lastile
    lastile(fullname, 'tiles', 1000, 20, debug=True)
    # 2. lasground
    lasground(head, 'tiles', 'ground', debug=True)
    # 3. remove_buffer
    remove_buffer(head, ('tiles', 'ground'), debug=True)
    # 4. lasmerge
    merged = lasmerge(head, ('tiles', 'ground', 'no_buffer'), head, 'nz14_gnd.laz')
    # 5. lasplit
    lassplit(merged, method='keep_class 2', debug=True)
    # 6. lasgrid
    gnd2 = os.path.join(head, '')
    grid = lasgrid(gnd2, 2, debug=True)
    # 7. las2las
    out = las2las(grid, debug=True)
    return out
