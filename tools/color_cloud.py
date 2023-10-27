import os

from ..config.config import cc_custom
from ..tools import misc


def color_cloud_with_tif(cloud, tif):

    out = os.path.splitext(cloud)[0] + '_rgb.laz'

    cmd = [cc_custom]
    cmd.extend(['-SILENT', '-NO_TIMESTAMP'])
    cmd.extend(['-C_EXPORT_FMT', 'LAS'])
    cmd.extend(['-AUTO_SAVE', 'OFF'])

    cmd.extend(['-O', '-GLOBAL_SHIFT', 'AUTO', cloud])
    cmd.extend(['-COORD_TO_SF', 'Z'])
    cmd.extend(['-SF_ADD_CONST', 'Height', '0'])
    cmd.extend(['-SF_TO_COORD', 'Height', 'Z'])

    cmd.extend(['-O', '-GLOBAL_SHIFT', 'FIRST', tif])

    cmd.append('-COLOR_INTERP')

    cmd.append('-POP_CLOUDS')  # remove the cloud related to the tif cloud

    cmd.extend(['-SF_TO_COORD', 'Coord. Z', 'Z'])

    cmd.extend(['-REMOVE_SF', 'Coord. Z'])
    cmd.extend(['-REMOVE_SF', 'Height'])

    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd, verbose=True)