# formerly known as interpolation_Poisson_v2 (Baptiste Feldmann)

import os
import shutil
import time

import numpy as np

import lidar_platform as lp
import tools.misc as utils


def poisson_recon_step1(filename, water_surface, bbox_place, tile_size,
                        params_recon, params_cc, params_normal):

    head, tail = os.path.split(filename)
    root, ext = os.path.splitext(tail)

    if not os.path.exists(water_surface):
        print('water_surface not found!')
        return None

    deb = time.time()

    normals_ply = filename[0:-4] + "_normals.ply"
    # compute normals if needed
    if not os.path.exists(normals_ply):
        lp.cloudcompare.compute_normals(filename, params_normal)
        lp.cloudcompare.last_file(filename[0:-4] + "_20*.ply",
                                  root + "_normals.ply")

    lp.cloudcompare.poisson(normals_ply, params_recon)

    # sample points on the mesh
    lp.cloudcompare.sample_mesh(
        lp.cloudcompare.open_file(params_cc, filename[0:-4] + "_normals_mesh.ply"),
        5)
    lp.cloudcompare.last_file(filename[0:-4] + "_normals_mesh_SAMPLED_POINTS_20*.laz",
                              root + "_sample_mesh.laz")

    # get the bounding box and clip the data
    lowerleft_x_lowerleft_y_size = np.array(
        root.split(sep="_")[bbox_place: bbox_place + 2] + [tile_size],
        dtype=str)
    print(f'filename {filename}, bbox {lowerleft_x_lowerleft_y_size}')
    lp.cloudcompare.las2las_keep_tile(filename[0:-4] + "_sample_mesh.laz", lowerleft_x_lowerleft_y_size)

    # compute distances between the points sampled on the mesh and the water surface
    query = lp.cloudcompare.open_file(params_cc, [filename[0:-4] + "_sample_mesh_1.laz", water_surface])
    lp.cloudcompare.c2c_dist(query, True, 10)
    lp.cloudcompare.last_file(filename[0:-4] + "_sample_mesh_1_C2C_DIST_20*.laz",
                              root + "_sample_mesh_1_C2C.laz")
    temp = lp.cloudcompare.last_file(water_surface[0:-4] + "_20*.laz")
    os.remove(temp)

    # keep only points which are below the water surface and not to far from it
    data = lp.lastools.read(filename[0:-4] + "_sample_mesh_1_C2C.laz", extra_fields=True)
    select = (data.c2c__absolute__distances < 100) & \
             (-10 < data.c2c__absolute__distances__z) & (data.c2c__absolute__distances__z < -1)

    if any(select):
        # store the filtered points
        data_new = lp.lastools.filter_las(data, select)
        lp.lastools.WriteLAS(filename[0:-4] + "_sample_mesh_1_1.laz", data_new)

        # compute cloud to cloud distances between the remaining points and the initial points
        query = lp.cloudcompare.open_file(params_cc, [filename[0:-4] + "_sample_mesh_1_1.laz", filename])
        lp.cloudcompare.c2c_dist(query, False, 10)
        lp.cloudcompare.last_file(filename[0:-4] + "_sample_mesh_1_1_C2C_DIST_20*.laz",
                                  root + "_sample_mesh_1_1_C2C.laz")
        temp = lp.cloudcompare.last_file(filename[0:-4] + "_20*.laz")
        os.remove(temp)

        # keep points which are close to the points sampled on the mesh... but not too much!
        data = lp.lastools.read(filename[0:-4] + "_sample_mesh_1_1_C2C.laz", extra_fields=True)
        select = (0.5 < data.c2c__absolute__distances) & ( data.c2c__absolute__distances < 200)
        os.remove(filename[0:-4] + "_sample_mesh_1_1_C2C.laz")
        if any(select):
            data_new = lp.lastools.filter_las(data, select)
            lp.lastools.WriteLAS(filename[0:-4] + "_sample_mesh_step1.laz", data_new)
        else:
            print("No point")
        os.remove(filename[0:-4] + "_sample_mesh_1_1.laz")
    else:
        print("No point")

    # clean temporary files
    os.remove(normals_ply)
    os.remove(filename[0:-4] + "_normals_mesh.ply")
    os.remove(filename[0:-4] + "_sample_mesh_1.laz")
    os.remove(filename[0:-4] + "_sample_mesh_1_C2C.laz")

    print("Duration : %.1f sec\n" % (time.time() - deb))


def poisson_recon_step2(filename, surface_water, bbox_cut,
                        params_interp, params_cc, params_normal):

    # bbox_cut=[minX,minY,maxX,maxY]

    if not os.path.exists(filename):
        raise Exception(f'file does not exists: {filename}')

    print(f'[poisson_recon_step2] process {filename}')

    head, tail = os.path.split(filename)
    root, ext = os.path.splitext(tail)
    
    deb = time.time()
    
    # compute normals using CloudCompare
    lp.cloudcompare.compute_normals(filename, params_normal)
    lp.cloudcompare.last_file(filename[0:-4] + "_20*.ply", 
                              root + "_normals.ply")
    
    # compute poisson reconstruction using CloudCompare
    lp.cloudcompare.poisson(filename[0:-4] + "_normals.ply", params_interp)
    
    # sample points on the mesh
    lp.cloudcompare.sample_mesh(
        lp.cloudcompare.open_file(params_cc, filename[0:-4] + "_normals_mesh.ply"),
        5)
    lp.cloudcompare.last_file(filename[0:-4] + "_normals_mesh_SAMPLED_POINTS_20*.laz",
                              root + "_sample_mesh.laz")

    # clip the data
    lp.cloudcompare.las2las_clip_xy(filename[0:-4] + "_sample_mesh.laz", bbox_cut)

    # compute cloud to cloud distances between the points sampled on the mesh and the water surface
    query = lp.cloudcompare.open_file(params_cc,
                                      [filename[0:-4] + "_sample_mesh_1.laz", surface_water])
    lp.cloudcompare.c2c_dist(query, True, 10)
    lp.cloudcompare.last_file(filename[0:-4] + "_sample_mesh_1_C2C_DIST_20*.laz",
                              root + "_sample_mesh_1_C2C.laz")
    temp = lp.cloudcompare.last_file(surface_water[0:-4] + "_20*.laz")
    os.remove(temp)
    
    # filter points by the distance to the water surface
    data = lp.lastools.read(filename[0:-4] + "_sample_mesh_1_C2C.laz", extra_fields=True)
    select = (data.c2c__absolute__distances < 100) & \
             (-10 < data.c2c__absolute__distances__z) & (data.c2c__absolute__distances__z < -1)
    os.remove(filename[0:-4] + "_sample_mesh_1_C2C.laz")
    
    if any(select):
        # store the filtered points
        data_new = lp.lastools.filter_las(data, select)
        lp.lastools.WriteLAS(filename[0:-4] + "_sample_mesh_1_1.laz", data_new)

        # compute cloud to cloud distances between the filtered points and the initial points
        query = lp.cloudcompare.open_file(params_cc,
                                          [filename[0:-4] + "_sample_mesh_1_1.laz", filename])
        lp.cloudcompare.c2c_dist(query, False, 10)
        lp.cloudcompare.last_file(filename[0:-4] + "_sample_mesh_1_1_C2C_DIST_20*.laz",
                                  filename[0:-4] + "_sample_mesh_1_1_C2C.laz")
        temp = lp.cloudcompare.last_file(filename[0:-4] + "_20*.laz")
        os.remove(temp)

        # keep points which are close to the points sampled on the mesh
        data = lp.lastools.read(filename[0:-4] + "_sample_mesh_1_1_C2C.laz", extra_fields=True)
        select = (0.5 < data.c2c__absolute__distances) & (data.c2c__absolute__distances < 200)
        os.remove(filename[0:-4] + "_sample_mesh_1_1_C2C.laz")
        if any(select):
            data_new = lp.lastools.filter_las(data, select)
            lp.lastools.WriteLAS(filename[0:-4] + "_sample_mesh_final.laz", data_new)
        else:
            print("No points")
        os.remove(filename[0:-4] + "_sample_mesh_1_1.laz")
    else:
        print("No points")
        
    # remove unused files
    os.remove(filename[0:-4] + "_normals_mesh.ply")
    os.remove(filename[0:-4] + "_normals.ply")
    os.remove(filename[0:-4] + "_sample_mesh.laz")
    os.remove(filename[0:-4] + "_sample_mesh_1.laz")
    
    print("Duration : %.1f sec\n" % (time.time() - deb))


def get_4_connected_neighbors(coords, tile_size):
    # return XY lower left coordinates of all neighbors in 4-connect
    # coords=[X,Y] lower left
    x = int(coords[0])
    y = int(coords[1])
    neighbor_coordinates = {"left": [str(x - tile_size), str(y)],
                            "up": [str(x), str(y + tile_size)],
                            "right": [str(x + tile_size), str(y)],
                            "down": [str(x), str(y - tile_size)]}
    return neighbor_coordinates


def get_info_from_filename(filename, bbox_place):
    root, ext = os.path.splitext(filename)
    split_filename = root.split(sep="_")
    prefix = split_filename[: bbox_place]
    coords = split_filename[bbox_place: bbox_place + 2]
    suffix = split_filename[bbox_place + 2:]
    return prefix, coords, suffix


def listing_neighbors(filenames, bbox_place, tile_size, extension='.laz'):
    # bbox_place
    # tile_size
    dict_ = {}
    for filename in filenames:
        prefix, coords, suffix = get_info_from_filename(filename, bbox_place)
        dict_of_neighbors = get_4_connected_neighbors(coords, tile_size)
        dict_[filename] = dict(zip(dict_of_neighbors.keys(), ["", "", "", ""]))
        for neighbor, coordinates in dict_of_neighbors.items():
            neighbor_name = "_".join(
                prefix +
                coordinates +
                suffix) + extension
            if neighbor_name in filenames and neighbor_name not in dict_.keys():
                dict_[filename][neighbor] = neighbor_name
    return dict_


def bbox_to_cut(coords, position, tile_size, dist_cut, buffer=0):
    # coords=[X, Y] lower left point
    # neighbor ('left', 'right', 'up', 'down')
    # tile_size
    # dist_cut
    # dictionary of the neighbor tiles lower left / upper right points
    # [minX, minY, maxX, maxY]
    dict_ = {"left": [-dist_cut,
                      -buffer,
                      dist_cut,
                      tile_size + buffer],
             "right": [tile_size - dist_cut,
                       -buffer,
                       tile_size + dist_cut,
                       tile_size + buffer],
             "up": [-buffer,
                    tile_size - dist_cut,
                    tile_size + buffer,
                    tile_size + dist_cut],
             "down": [-buffer,
                      -dist_cut,
                      tile_size + buffer,
                      dist_cut]}
    # initialize bbox with the incoming coordinates
    bbox = np.array([coords[0], coords[1], coords[0], coords[1]])
    bbox += np.array(dict_[position])
    return np.array(bbox, dtype=int)


def poisson_recon(filename, water_surface, bbox_place, tile_size,
                  params_recon, params_cc, params_normal,
                  z_ws_min=-10, z_ws_max=-1, d_ws_max=100,
                  d_orig_min=0.5, d_orig_max=100, buffer=None):

    head, tail = os.path.split(filename)
    root, ext = os.path.splitext(tail)
    odir = os.path.join(head, 'PoissonRecon')
    os.makedirs(odir, exist_ok=True)

    if not os.path.exists(water_surface):
        print('water_surface not found!')
        return None

    deb = time.time()

    if buffer:  # clip the data before performing the Poisson reconstruction
        # get the bounding box and clip the data keeping a given buffer
        lowerleft_x_lowerleft_y_size = np.array(
            root.split(sep="_")[bbox_place: bbox_place + 2] + [tile_size],
            dtype=str)
        x_min, y_min, size = lowerleft_x_lowerleft_y_size.astype(float)
        filename_cut = filename[0:-4] + "_cut.laz"
        query = "las2las -i " + filename + \
                f" -keep_tile {x_min - buffer} {y_min - buffer} {tile_size + 2 * buffer}" + \
                " -o " + filename_cut
        utils.run_bis(query)

    # compute normals if needed
    normals_ply = filename[0:-4] + "_normals.ply"
    if buffer:
        lp.cloudcompare.compute_normals(filename_cut, params_normal)
        lp.cloudcompare.last_file(filename_cut[0:-4] + "_20*.ply",
                                  root + "_normals.ply")
    else:
        lp.cloudcompare.compute_normals(filename, params_normal)
        lp.cloudcompare.last_file(filename[0:-4] + "_20*.ply",
                                  root + "_normals.ply")

    # compute poisson reconstruction using PoissonRecon.exe
    lp.cloudcompare.poisson(normals_ply, params_recon)

    # sample points on the mesh
    lp.cloudcompare.sample_mesh(
        lp.cloudcompare.open_file(params_cc, filename[0:-4] + "_normals_mesh.ply"),
        5)
    lp.cloudcompare.last_file(filename[0:-4] + "_normals_mesh_SAMPLED_POINTS_20*.laz",
                              root + "_sample_mesh.laz")

    # compute distances between the points sampled on the mesh and the water surface
    query = lp.cloudcompare.open_file(params_cc, [filename[0:-4] + "_sample_mesh.laz", water_surface])
    lp.cloudcompare.c2c_dist(query, xyz=True, octree_lvl=10)
    lp.cloudcompare.last_file(filename[0:-4] + "_sample_mesh_C2C_DIST_20*.laz",
                              root + "_sample_mesh_C2C_ws.laz")
    temp = lp.cloudcompare.last_file(water_surface[0:-4] + "_20*.laz")
    os.remove(temp)

    # keep only points which are below the water surface and not to far from it
    data = lp.lastools.read(filename[0:-4] + "_sample_mesh_C2C_ws.laz", extra_fields=True)
    select = (data.c2c__absolute__distances < d_ws_max) & \
             (z_ws_min < data.c2c__absolute__distances__z) & (data.c2c__absolute__distances__z < z_ws_max)

    if any(select):
        # store the filtered points
        data_new = lp.lastools.filter_las(data, select)
        lp.lastools.WriteLAS(filename[0:-4] + "_sample_mesh_select0.laz", data_new)

        # compute cloud to cloud distances between the remaining points and the initial points
        query = lp.cloudcompare.open_file(params_cc, [filename[0:-4] + "_sample_mesh_select0.laz", filename])
        lp.cloudcompare.c2c_dist(query, False, 10)
        lp.cloudcompare.last_file(filename[0:-4] + "_sample_mesh_select0_C2C_DIST_20*.laz",
                                  root + "_sample_mesh_select0_C2C_orig.laz")
        temp = lp.cloudcompare.last_file(filename[0:-4] + "_20*.laz")
        os.remove(temp)
        os.remove(filename[0:-4] + "_sample_mesh_select0.laz")

        # keep sampled points which are close to the original points... but not too much!
        data = lp.lastools.read(filename[0:-4] + "_sample_mesh_select0_C2C_orig.laz", extra_fields=True)
        select = (d_orig_min < data.c2c__absolute__distances) & (data.c2c__absolute__distances < d_orig_max)
        os.remove(filename[0:-4] + "_sample_mesh_select0_C2C_orig.laz")
        if any(select):
            data_new = lp.lastools.filter_las(data, select)
            lp.lastools.WriteLAS(filename[0:-4] + "_sample_mesh_select1.laz", data_new)

            # get the bounding box and clip the data
            lowerleft_x_lowerleft_y_size = np.array(
                root.split(sep="_")[bbox_place: bbox_place + 2] + [tile_size],
                dtype=str
            )
            print(f'clip {filename} to bbox {lowerleft_x_lowerleft_y_size}')
            lp.cloudcompare.las2las_keep_tile(filename[0:-4] + "_sample_mesh_select1.laz", lowerleft_x_lowerleft_y_size)
            os.remove(filename[0:-4] + "_sample_mesh_select1.laz")
        else:
            print("No point")
    else:
        print("No point")

    # clean temporary files
    os.remove(filename[0:-4] + "_normals.ply")
    os.remove(filename[0:-4] + "_normals_mesh.ply")
    os.remove(filename[0:-4] + "_sample_mesh.laz")
    os.remove(filename[0:-4] + "_sample_mesh_C2C_ws.laz")

    print("Duration : %.1f sec\n" % (time.time() - deb))

    src = filename[0:-4] + "_sample_mesh_select1_1.laz"
    if os.path.exists(src):
        dst = shutil.move(src, odir)
        return dst
    else:
        return None
