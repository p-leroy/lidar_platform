import glob, os

import laspy

idir = r'C:\DATA\Brioude_30092021\05-Traitements\C2\denoised'
pattern = '*_C2_r_1.laz'

files = glob.glob(os.path.join(idir, pattern))

N = 0
n_files = 0
for file in files:
    print(file)
    a = laspy.open(file)
    n = a.header.point_count
    N += n
    n_files += 1

print(f'total number of points {N}, number of files {n_files}')
