from lidar_platform import cc
from lidar_platform.config.config import cc_custom

file = r'C:\DATA\TMP\10.bin'
file_laz = r'C:\DATA\TMP\C2_HD.laz'
file_e57 = r'C:\DATA\TMP\Mangarere_FARO_20170314_13.e57'

#%% -DISTANCES_FROM_SENSOR
out = cc.distances_from_sensor(file, squared=True, verbose=True, cc_exe=cc_custom)

#%% -SCATTERING_ANGLE
out = cc.scattering_angles(file, degrees=True, verbose=True, cc_exe=cc_custom)

#%% -SS
out = cc.ss(file, 'SPATIAL', 1, verbose=True)

#%% -OCTREE_NORMAL
out = cc.octree_normals(file_e57, 0.1, with_grids=True, angle=1, orient='WITH_GRIDS',
                        silent=True, verbose=True, global_shift='AUTO',
                        cc='C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe')

#%% -SF_ADD_CONST
to_add = (("spam", 0.5),
          ("eggs", 5))
out = cc.sf_add_const(file_laz, to_add, in_place=False,
                      silent=True, verbose=True, fmt='sbf')
