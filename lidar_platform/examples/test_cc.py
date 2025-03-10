from lidar_platform import cc

file = r'C:\DATA\TMP\10.bin'

#%%
out = cc.distances_from_sensor(file, squared=True)

#%%
out = cc.scattering_angles(file, degrees=True)

#%%
out = cc.ss(file, 'SPATIAL', 1, verbose=True)

#%%
