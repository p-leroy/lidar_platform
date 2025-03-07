# M3C2 is a CloudCompare plugin

import configparser
import os


def set_search_scale(filename, search_scale):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(filename)
    config['General']['SearchScale'] = str(search_scale)
    config.write(filename)


def get_search_scale(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    config = configparser.ConfigParser()
    config.read(filename)
    return eval(config['General']['SearchScale'])