# coding: utf-8
# Baptiste Feldmann, Paul Leroy
# CloudCompare calls

import glob
import os

from joblib import Parallel, delayed

from ..config.config import EXPORT_FMT, SHIFT, QUERY_0
from ..tools import misc

# Poisson reconstruction parameters
POISSON_RECON_PARAMETERS = {"bType": {"Free": "1", "Dirichlet": "2", "Neumann": "3"}}


def c2c_dist(query, xyz=True, octree_lvl=0):
    """Run C2C distance

    Args:
        query (str): CC query
        xyz (bool, optional): save X,Y and Z distance. Defaults to True.
        octree_lvl (int, optional): force CC to a specific octree level,
        useful when extent of two clouds are very different,
            0 means that you let CC decide. Defaults to 0.
    """
    if xyz:
        opt_xyz = "-split_xyz"
    else:
        opt_xyz = ""
        
    if octree_lvl == 0:
        opt_octree = ""
    else:
        opt_octree = " -octree_level " + str(octree_lvl)

    if "-fwf_o" in query:
        opt_save = " -fwf_save_clouds"
    else:
        opt_save = " -save_clouds"
        
    misc.run(query + " -C2C_DIST " + opt_xyz + opt_octree + opt_save)


def c2c_files(params, files, filepath_b, octree_lvl=9, nbr_job=5):
    """Run C2C distance between several pointClouds and a specific pointCloud

    Args:
        params (list): CC parameter [QUERY_0, Export_fmt, global_shift]
        files (list): list of files in workspace
        filepath_b (str): file to which the distance is computed
        octree_lvl (int, optional): force CC to a specific octree level,
        useful when extent of two clouds are very different,
            0 means that you let CC decide. Defaults to 9.
        nbr_job (int, optional): The number of jobs to run in parallel. Defaults to 5.
    """

    if files is []:
        print("[c2c_files] no file to process, stop")
        return
    else:
        print("[c2c_files] %i files to process" % len(files))

    # compute cloud to cloud distances
    list_query = []
    for file in files:
        list_query += [open_file(params, [file, filepath_b])
                       + " -C2C_DIST -split_xyz -octree_level " + str(octree_lvl) + " -save_clouds"]
    Parallel(n_jobs=nbr_job, verbose=0)(delayed(misc.run)(cmd) for cmd in list_query)

    # clean temporary files
    for file in files:
        c2c_dist_files = file[0:-4] + "_C2C_DIST_*.laz"
        last_file(c2c_dist_files, file[0:-4] + "_C2C.laz")
        for to_delete in glob.glob(c2c_dist_files):
            os.remove(to_delete)
        
    for file in glob.glob(filepath_b[0:-4] + "_20*.laz"):
        os.remove(file)

    print("[c2c_files] done")


def c2m_dist(command, max_dist=0, octree_lvl=0, cores=0):
    """
    Cloud-to-Mesh distances between the first cloud (compared) and the first loaded mesh (reference).
    """
    opt = ""
    if max_dist > 0:
        opt += " -max_dist " + str(max_dist)
    
    if octree_lvl > 0:
        opt += " -octree_level " + str(octree_lvl)

    if cores > 0:
        opt += " -max_tcount " + str(cores)

    misc.run(command + " -C2M_DIST" + opt + " -save_clouds")


def las2las_keep_tile(filepath, lowerleft_x_lowerleft_y_size):
    # lowerleft_x_lowerleft_y_size = [lowerleft_x, lowerleft_y, size]
    # keeps a size by size tile with a lower left coordinate of x=lowerleft_x and y=lowerleft and stores as a
    # compressed LAZ file *_1.laz (an added '_1' in the name).
    query = "las2las -i " + filepath + " -keep_tile " + " ".join(lowerleft_x_lowerleft_y_size) + " -odix _1 -olaz"
    misc.run(query)


def las2las_clip_xy(filepath, min_x_min_y_max_x_max_y):
    # min_x_min_y_max_x_max_y = [min_x, min_y, max_x, max_y]
    query = "las2las -i " + filepath + " -keep_xy " + " ".join(min_x_min_y_max_x_max_y) + " -odix _1 -olaz"
    misc.run(query)


def compute_normals(filepath, params):
    """Compute normal components and save it in PLY file format

    Args:
        filepath (str): path to input LAS file
        params (list): CC parameters [shiftname, normalRadius,model (LS / TRI / QUADRIC)]
    """    
    query = open_file(["standard", "PLY_cloud", params["shiftname"]], filepath)
    misc.run(query +
                  " -octree_normals " + params["normal_radius"] +
                  " -orient PLUS_Z -model " + params["model"] +
                  " -save_clouds")


def compute_normals_dip(filepath, cc_param, radius, model="LS"):
    """Compute normals and save 'dipDegree' attribute in LAS file

    Args:
        filepath (str): path ot input LAS file
        cc_param (list): CC parameters [QUERY_0,Export_fmt,shiftname]
        radius (float): 
        model (str, optional): local model type LS / TRI / QUADRIC. Defaults to "LS".
    """
    query = open_file(cc_param, filepath)
    misc.run(query +
                  " -octree_normals " + str(radius) +
                  " -orient PLUS_Z -model " + model +
                  " -normals_to_dip -save_clouds")


def compute_feature(query, features_dict):
    for i in features_dict.keys():
        query += " -feature " + i + " " + str(features_dict[i])
    
    misc.run(query + " -save_clouds")


def create_raster(command, grid_size, interp=False):
    """
    CloudCompare command for rasterization
    """
    command += " -rasterize -grid_step " + str(grid_size) + " -vert_dir 2 -proj AVG -SF_proj AVG"
    if interp:
        command += " -empty_fill INTERP"

    command += " -output_raster_z -save_clouds"
    misc.run(command)


def density(command, radius):
    """
    CloudCompare command for density computation
    """
    command += " -density " + str(radius) + " -type KNN -save_clouds"
    misc.run(command)


def last_file(filepath, new_name=None, verbose=False):
    """return and modify last file created according to a given pattern

    Args:
        filepath (str): pattern of searched file ex: D:/travail/*_lastfile.las
        new_name (str, optional): new name to searched file,
            if new_name=str : rename searched file and return new path
            otherwise : return path of searched file.
            Defaults to None.
        verbose

    Returns:
        str: path of searched file
    """
    files = glob.glob(filepath)
    time = []
    for file in files:
        time += [os.path.getmtime(file)]
    src = files[time.index(max(time))]
    head, tail = os.path.split(src)
    if new_name is not None:
        dst = os.path.join(head, new_name)
        if os.path.exists(dst):
            os.remove(dst)
            if verbose:
                print(f'remove file before renaming: {dst}')
        os.rename(src, dst)
        if verbose:
            print(f'rename {src} => {dst}')
        return dst
    else:
        return src


def merge_clouds(command):
    """
    CloudCompare command for merging clouds
    """
    if "-fwf_o" in command:
        opt1 = "-fwf_save_clouds"
    else:
        opt1 = "-save_clouds"
    
    command += " -merge_clouds " + opt1
    misc.run(command)


def m3c2(query, params_file):
    """Run M3C2 plugin

    Args:
        query (str): CC query
        params_file (str): path to M3C2 parameter textfile
    """
    query += " -M3C2 " + params_file
    misc.run(query)


def open_file(params, filepath, fwf=False):
    """Construct CC query to open file

    Args:
        params (list): CC parameter [Query0,Export_fmt,shiftname]
        filepath (str or list of string): path to input file or list of input files
        fwf (bool, optional): True if you want to open LAS file with full-waveform. Defaults to False.

    Raises:
        TypeError: filepath must be str or list type

    Returns:
        str: CC query
    """

    if fwf:
        opt_fwf = " -fwf_o"
    else:
        opt_fwf = " -O"
    
    query = QUERY_0[params[0]] + EXPORT_FMT[params[1]]
    if type(filepath) is list:
        for i in filepath:
            query += opt_fwf + " -global_shift " + SHIFT[params[2]] + " " + i
    elif type(filepath) is str:
        query += opt_fwf + " -global_shift " + SHIFT[params[2]] + " " + filepath
    else:
        raise TypeError("filepath must be a string or a list of string !")
        
    return query


def ortho_wfm(query, param_file):
    """Run ortho-waveform plugin

    Args:
        query (str): CC query
        param_file (str): ortho-waveform plugin textfile parameter
    """
    query += " -fwf_ortho "+param_file+" -fwf_save_clouds"
    misc.run(query)


def wfw_peaks(query, param_file):
    """Run ortho-waveform plugin and find peaks

    Args:
        query (str): 
        param_file (str): ortho-waveform plugin find peaks textfile parameter
    """
    query += " -fwf_peaks " + param_file + " -fwf_save_clouds"
    misc.run(query)


def sf_grad(command, sf_index):
    """
    CloudCompare command for the calculation of the gradient of a scalar field
    """
    command += " -set_active_sf " + str(sf_index) + " -SF_grad TRUE -save_clouds"
    misc.run(command)


def poisson(filename, params):
    """Run Poisson Surface Reconstruction
    See docs https://www.cs.jhu.edu/~misha/Code/PoissonRecon/
    
    Args:
        filename (str): path ot input PLY file
        params (dict): parameter dictionary
            ex: {"bType":"Neumann","degree":"2",...}
    """
    query = QUERY_0['PoissonRecon'] + " --in " + filename + " --out " + filename[0:-4] + "_mesh.ply"
    for i in params.keys():
        query += " --" + i + int(bool(len(params[i]))) * " "
        if i in POISSON_RECON_PARAMETERS.keys():
            query += POISSON_RECON_PARAMETERS[i][params[i]]
        else:
            query += params[i]
    print(f'[poisson] query: {query}')
    misc.run(query)


def rasterize(command, grid_size, proj, empty):
    """
    CloudCompare command for rasterization
    """
    command += " -rasterize -grid_step " + str(grid_size) + " -vert_dir 2 -proj " \
               + proj + " -SF_proj " + proj
    if empty == "empty":
        command += " -output_cloud -save_clouds"
    else:
        command += " -empty_fill " + empty + \
                   " -output_cloud -save_clouds"
    misc.run(command)


def sample_mesh(query, density):
    query += f" -sample_mesh DENSITY {density} -save_clouds"
    misc.run(query)


def filter_sf(command, sf_index, mini, maxi):
    """
    CC command for SF filtering
    """
    command += " -set_active_sf " + str(sf_index) + " -filter_sf " \
               + str(mini) + " " + str(maxi) + " -save_clouds"
    misc.run(command)


def subsampling(command, min_dist):
    command += " -SS SPATIAL " + str(min_dist) + " -save_clouds"
    misc.run(command)
