import os, logging, shutil

from . import utils


logger = logging.getLogger(__name__)
logging.basicConfig()

class check(object):
    def __init__(self):
        self.check_utils()

    def _utils_exception(self, value):
        logger.warning(f" path invalid: {value}")

    def check_utils(self):
        for key, value in utils.QUERY_0.items():
            if not os.path.exists(value.split(" ")[0]+".exe"):
                self._utils_exception(value)

        if shutil.which(utils.GDAL_QUERY_ROOT.split(" ")[0]) is None:
            self._utils_exception("GDAL_QUERY_ROOT")

        if not os.path.isdir(utils.VERTICAL_DATUM_DIR):
            self._utils_exception("VERTICAL_DATUM_DIR")

    
