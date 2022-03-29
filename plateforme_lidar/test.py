from . import utils
import os,shutil

class check(object):
    def __init__(self):
        self.check_utils()

    def _utils_exception(self,constant):
        raise Exception("<plateforme_lidar.utils."+constant+" : path invalid>")

    def check_utils(self):
        for key in utils.QUERY_0.keys():
            if not os.path.exists(utils.QUERY_0[key].split(" ")[0]+".exe"):
                self._utils_exception("QUERY_0")

        if shutil.which(utils.GDAL_QUERY_ROOT.split(" ")[0]) is None:
            self._utils_exception("GDAL_QUERY_ROOT")

        if not os.path.isdir(utils.VERTICAL_DATUM_DIR):
            self._utils_exception("VERTICAL_DATUM_DIR")

    
