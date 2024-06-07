import glob
import os

import laspy


def get_number_of_points(idir, pattern):
    """
    Get the number of points and the number of files
    :param idir: the directory containing the files
    :param pattern: the pattern used to build the list of files, e.g. '*_C2_r_1.laz'
    :return: (number of points, number of files)
    """

    files = glob.glob(os.path.join(idir, pattern))
    n_files = len(files)

    N = 0
    for file in files:
        print(file)
        a = laspy.open(file)
        n = a.header.point_count
        N += n

    print(f'total number of points {N}, number of files {n_files}')

    return N, n_files
