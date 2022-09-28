# formerly known as compute_density_on_PC.py (Baptiste Feldmann)

import numpy as np
from scipy.spatial import cKDTree

import lidar_platform as lp


def get_number_of_points_inside_radius(points, grid=None, radius=1., p_norm=2.):
    tree = cKDTree(points, leafsize=1000)

    if grid is None:
        grid = np.copy(points)

    return tree.query_ball_point(grid, r=radius, p=p_norm, return_length=True)


def define_grid(step, size_x, size_y, lower_left):
    x0, y0 = lower_left
    tab = []
    for i in range(x0, x0 + size_x, step):
        for c in range(y0, y0 + size_y, step):
            tab += [[i + 0.5 * step, c + 0.5 * step]]
    return np.array(tab)


def func(filepath, grid_step=1, radius=0.5):
    print(f'[density.func] process {filepath}')
    data = lp.lastools.read(filepath)
    lower_left = np.int_(np.amin(data.XYZ[:, 0:2], axis=0))
    bbox = np.int_(np.amax(data.XYZ[:, 0:2], axis=0)) - lower_left
    grid = define_grid(grid_step, bbox[0], bbox[1], lower_left)
    result = get_number_of_points_inside_radius(data.XYZ[:, 0:2], grid=grid, radius=radius, p_norm=np.inf)
    np.savez_compressed(filepath[0:-4] + "_density.npz", result)
    print(f'[density.func] done {filepath}')
