import os

from ...config.config import cc_exe
from .. import misc

from .CCCommand import CCCommand


def distances_from_sensor(pc, squared=False,
                          silent=True, verbose=False, global_shift='AUTO', fmt='bin',
                          cc_exe=cc_exe):

    root, ext = os.path.splitext(pc)
    out = root + '_RANGES.' + fmt.lower()

    cmd = CCCommand(cc_exe, silent=silent, fmt=fmt)  # create initial list
    cmd.open_file(pc, global_shift=global_shift)  # open files

    cmd.append('-DISTANCES_FROM_SENSOR')  # compute distances from sensor
    if squared:
        cmd.append('-SQUARED')

    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd, verbose=verbose)

    return out


def scattering_angles(pc, degrees=False,
                      silent=True, verbose=False, global_shift='AUTO', fmt='bin',
                      cc_exe=cc_exe):

    root, ext = os.path.splitext(pc)
    out = root + '_ANGLES.' + fmt.lower()

    cmd = CCCommand(cc_exe, silent=silent, fmt=fmt)
    cmd.open_file(pc, global_shift=global_shift)

    cmd.append('-SCATTERING_ANGLES')  # compute scattering angles
    if degrees:
        cmd.append('-DEGREES')

    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd, verbose=verbose)

    return out


def distances_from_sensor_and_scattering_angles(pc, squared=False, degrees=False,
                                                silent=True, verbose=False, global_shift='AUTO', fmt='bin',
                                                cc_exe=cc_exe):

    root, ext = os.path.splitext(pc)
    out = root + '_RANGES_ANGLES.' + fmt.lower()

    cmd = CCCommand(cc_exe, silent=silent, fmt='SBF')
    cmd.open_file(pc, global_shift=global_shift)

    cmd.append('-DISTANCES_FROM_SENSOR')  # compute distances from sensor
    if squared:
        cmd.append('-SQUARED')

    cmd.append('-SCATTERING_ANGLES')  # compute scattering angles
    if degrees:
        cmd.append('-DEGREES')

    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd, verbose=verbose)

    return out
