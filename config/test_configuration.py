#!python

import logging
import os
import shutil

from tools import misc

logger = logging.getLogger(__name__)
logging.basicConfig()


def utils_exception(constant):
        raise Exception("<plateforme_lidar.utils." + constant + " : path invalid>")


def exists(path):
    if not os.path.exists(path):
        utils_exception(path)
    else:
        print(f'   => {path} exists')


def check_utils():
    print("check 'standard_view'")
    exists(utils.QUERY_0["standard_view"])
    print("check 'cc_ple_view'")
    exists(utils.QUERY_0["cc_ple_view"])
    print("check 'PoissonRecon'")
    exists(utils.QUERY_0["PoissonRecon"])

    print("check GDAL_QUERY_ROOT")
    to_test = utils.GDAL_QUERY_ROOT
    if shutil.which(to_test.split(" ")[0]) is None:
        utils_exception(to_test)
    else:
        print("   => valid path: " + to_test)

    print("check VERTICAL_DATUM_DIR")
    to_test = utils.VERTICAL_DATUM_DIR
    if not os.path.isdir(to_test):
        utils_exception(to_test)
    else:
        print("   => valid path: " + to_test)


if __name__ == '__main__':
    check_utils()
