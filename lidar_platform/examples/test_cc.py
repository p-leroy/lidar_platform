from lidar_platform import cc
from lidar_platform.config.config import cc_custom

file = r'C:\DATA\TMP\10.bin'
file_e57 = r'C:\DATA\TMP\Mangarere_FARO_20170314_13.e57'

#%%
out = cc.distances_from_sensor(file, squared=True, verbose=True, cc_exe=cc_custom)

#%%
out = cc.scattering_angles(file, degrees=True, verbose=True, cc_exe=cc_custom)

#%%
out = cc.ss(file, 'SPATIAL', 1, verbose=True)

#%%
out = cc.octree_normals(file_e57, 0.1, with_grids=True, angle=1, orient='WITH_GRIDS',
                        silent=True, verbose=True, global_shift='AUTO',
                        cc='C:/opt/CloudCompareProjects/CloudCompare/CloudCompare.exe')
