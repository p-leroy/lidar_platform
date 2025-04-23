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
filename = r"C:\DATA\TMP\Normals\Mangarere_2014003.OctreeNormal.bin"
out = cc.octree_normals(filename, 5, with_grids=True, angle=1, orient='PREVIOUS',
                        silent=False, verbose=True, global_shift='AUTO',
                        cc='C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe')

#%% -OCTRE_NORMAL
filename = r"C:\DATA\TMP\Normals\Mangarere_20120227_RegV2_S1_SMALL.e57"
out = cc.octree_normals(filename, 0.1, with_grids=False, angle=1, orient='WITH_SENSOR',
                        silent=False, verbose=True, fmt='e57', global_shift='AUTO',
                        all_at_once=True,
                        cc='C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe')

#%% -SF_ADD_CONST
to_add = (("spam", 0.5),
          ("eggs", 5))
out = cc.sf_add_const(file_laz, to_add, in_place=False,
                      silent=True, verbose=True, fmt='sbf')
