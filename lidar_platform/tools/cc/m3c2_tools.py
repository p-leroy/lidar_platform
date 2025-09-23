# M3C2 is a CloudCompare plugin

import configparser
import os


normals_computation_mode = {
    "DEFAULT_MODE": 0,  # compute normals on core points
    "USE_CLOUD1_NORMALS": 1,
    "MULTI_SCALE_MODE": 2,
    "VERT_MODE": 3,
    "HORIZ_MODE": 4,
    "USE_CORE_POINTS_NORMALS": 5,
}


def set_search_scale(filename, search_scale):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(filename)
    config['General']['SearchScale'] = str(search_scale)
    with open(filename, 'w') as f:
        config.write(f)


def get_search_scale(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    config = configparser.ConfigParser()
    config.read(filename)
    return eval(config['General']['SearchScale'])