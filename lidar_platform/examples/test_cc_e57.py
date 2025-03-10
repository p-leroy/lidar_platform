from lidar_platform import cc

file = r'C:\DATA\TMP\10.bin'

#%%
cc.distances_from_sensor(file, squared=True)

#%%
cc.scattering_angles(file, degrees=True)
