import os, logging, shutil

from . import utils


logger = logging.getLogger(__name__)
logging.basicConfig()

class check(object):
    def __init__(self):
        self.check_utils()

    def _utils_exception(self, value):
        logger.warning(f" path invalid: {value}")

    def exists(self, path):
        if not os.path.exists(path):
            self._utils_exception(path)

    def check_utils(self):

        self.exists(utils.QUERY_0["standard_view"])
        self.exists(utils.QUERY_0["cc_ple_view"])
        self.exists(utils.QUERY_0["PoissonRecon"])

        if shutil.which(utils.GDAL_QUERY_ROOT.split(" ")[0]) is None:
            self._utils_exception(utils.GDAL_QUERY_ROOT)

        if not os.path.isdir(utils.VERTICAL_DATUM_DIR):
            self._utils_exception(utils.VERTICAL_DATUM_DIR)

    
