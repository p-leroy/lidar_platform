# coding: utf-8
# Baptiste Feldmann
# Liste de fonctions utilisant CloudCompare (CC)
from . import utils
from joblib import Parallel, delayed
import glob,os

def c2c_dist(query,XYZ=True,octree_lvl=0):
    """Run C2C distance

    Args:
        query (str): CC query
        XYZ (bool, optional): save X,Y and Z distance. Defaults to True.
        octree_lvl (int, optional): force CC to a specific octree level, useful when extent of two clouds are very different,
            0 means that you let CC decide. Defaults to 0.
    """
    if XYZ :
        opt_xyz="-split_xyz"
    else:
        opt_xyz=""
        
    if octree_lvl==0:
        opt_octree=""
    else:
        opt_octree=" -octree_level "+str(octree_lvl)

    if "-fwf_o" in query:
        opt_save=" -fwf_save_clouds"
    else:
        opt_save=" -save_clouds"
        
    utils.Run_bis(query+" -C2C_DIST "+opt_xyz+opt_octree+opt_save)
    
def c2c_files(params,workspace,list_filenameA,filepathB,octree_lvl=9,nbr_job=5):
    """Run C2C distance between several pointClouds and a specific pointCloud

    Args:
        params (list): CC parameter [QUERY_0,Export_fmt,shiftname]
        workspace (str): directory path (ended by '/')
        list_filenameA (list): list of files in workspace
        filepathB (str): file to which the distance is computed
        octree_lvl (int, optional): force CC to a specific octree level, useful when extent of two clouds are very different,
            0 means that you let CC decide. Defaults to 9.
        nbr_job (int, optional): The number of jobs to run in parallel. Defaults to 5.
    """

    print("[Cloud2Cloud] %i files" %len(list_filenameA))
    list_query=[]
    for f in list_filenameA:
        list_query+=[open_file(params,[workspace+f,filepathB])+" -C2C_DIST -split_xyz -octree_level "+str(octree_lvl)+" -save_clouds"]

    Parallel(n_jobs=nbr_job, verbose=0)(delayed(utils.Run_bis)(cmd) for cmd in list_query)

    for i in list_filenameA:
        last_file(workspace+i[0:-4]+"_C2C_DIST_*.laz",i[0:-4]+"_C2C.laz")
        
    for f in glob.glob(filepathB[0:-4]+"_20*.laz"):
        os.remove(f)
    print("[Cloud2Cloud] Process done")

def c2m_dist(commande,max_dist=0,octree_lvl=0,cores=0):
    """
    Cloud-to-Mesh distances between the first cloud (compared) and the first loaded mesh (reference).
    """
    opt=""
    if max_dist>0:
        opt+=" -max_dist "+str(max_dist)
    
    if octree_lvl>0:
        opt+=" -octree_level "+str(octree_lvl)

    if cores>0:
        opt+=" -max_tcount "+str(cores)

    utils.Run(commande+" -C2M_DIST"+opt+" -save_clouds")

def clip_tile(filepath,bbox):
    #bbox=[minX,minY,size]
    query="las2las -i "+filepath+" -keep_tile "+" ".join(bbox)+" -odix _1 -olaz"
    utils.Run_bis(query)

def clip_xy(filepath,bbox):
    #bbox=[minX,minY,maxX,maxY]
    query="las2las -i "+filepath+" -keep_xy "+" ".join(bbox)+" -odix _1 -olaz"
    utils.Run_bis(query)

def compute_normals(filepath,params):
    """Compute normal components and save it in PLY file format

    Args:
        filepath (str): path to input LAS file
        params (list): CC parameters [shiftname,normalRadius,model (LS / TRI / QUADRIC)]
    """    
    query=open_file(["standard","PLY_cloud",params["shiftname"]],filepath)
    utils.Run_bis(query+" -octree_normals "+params["normal_radius"]+" -orient PLUS_Z -model "+params["model"]+" -save_clouds")

def compute_normals_dip(filepath,CC_param,radius,model="LS"):
    """Compute normals and save 'dipDegree' attribute in LAS file

    Args:
        filepath (str): path ot input LAS file
        CC_param (list): CC parameters [QUERY_0,Export_fmt,shiftname]
        radius (float): 
        model (str, optional): local model type LS / TRI / QUADRIC. Defaults to "LS".
    """
    query=open_file(CC_param,filepath)
    utils.Run_bis(query+" -octree_normals "+str(radius)+" -orient PLUS_Z -model "+model+" -normals_to_dip -save_clouds")

def compute_feature(query,features_dict):
    for i in features_dict.keys():
        query+=" -feature "+i+" "+str(features_dict[i])
    
    utils.Run_bis(query+" -save_clouds")
    
def create_raster(commande,grid_size,interp=False):
    """
    Commande CC pour le calcul de grille
    """
    commande+=" -rasterize -grid_step "+str(grid_size)+\
                " -vert_dir 2 -proj AVG -SF_proj AVG"
    if interp:
        commande+=" -empty_fill INTERP"

    commande+=" -output_raster_z -save_clouds"
    utils.Run(commande)  

def densite(commande,radius):
    """
    Commande CC pour le calcul de densité
    """
    commande+=" -density "+str(radius)+" -type KNN -save_clouds"
    utils.Run(commande)

def last_file(filepath,new_name=None):
    """return and modify last file created according to a given pattern

    Args:
        filepath (str): pattern of searched file ex: D:/travail/*_lastfile.las
        new_name (str, optional): new name to searched file,
            if new_name=str : rename searched file and return new path
            otherwise : return path of searched file.
            Defaults to None.

    Returns:
        str: path of searched file
    """
    liste=glob.glob(filepath)
    time=[]
    for i in liste:
        time+=[os.path.getmtime(i)]
    file=os.path.split(liste[time.index(max(time))])
    if new_name != None :
        os.rename(file[0]+"/"+file[1],file[0]+"/"+new_name)
        return file[0]+"/"+new_name
    else :
        return file[0]+"/"+file[1]

def merge_clouds(commande):
    """
    Commande CC pour la fusion de plusieurs fichiers
    """
    if "-fwf_o" in commande:
        opt1="-fwf_save_clouds"
    else:
        opt1="-save_clouds"
    
    commande+=" -merge_clouds "+opt1
    utils.Run(commande)

def m3c2(query,params_file):
    """Run M3C2 plugin

    Args:
        query (str): CC query
        params_file (str): path to M3C2 parameter textfile
    """
    query+=" -M3C2 "+params_file+" -save_clouds"
    utils.Run_bis(query)
    
def open_file(params,filepath,fwf=False):
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
        opt_fwf=" -fwf_o"
    else:
        opt_fwf=" -O"
    
    query=utils.QUERY_0[params[0]]+utils.EXPORT_FMT[params[1]]
    if type(filepath) is list:
        for i in filepath:
            query+=opt_fwf+" -global_shift "+utils.SHIFT[params[2]]+" "+i
    elif type(filepath) is str:
        query+=opt_fwf+" -global_shift "+utils.SHIFT[params[2]]+" "+filepath
    else:
        raise TypeError("filepath must be a string or a list of string !")
        
    return query

def ortho_wavefm(query,param_file):
    """Run ortho-waveform plugin

    Args:
        query (str): CC query
        param_file (str): ortho-waveform plugin textfile parameter
    """
    query+=" -fwf_ortho "+param_file+" -fwf_save_clouds"
    utils.Run(query)

def peaks(query,param_file):
    """Run ortho-waveform plugin and find peaks

    Args:
        query (str): 
        param_file (str): ortho-waveform plugin find peaks textfile parameter
    """
    query+=" -fwf_peaks "+param_file+" -fwf_save_clouds"
    utils.Run(query)

def pente(commande,indexSF):
    """
    Commande CC pour le calcul de pente (gradient de la composante Z)
    """
    commande+=" -set_active_sf "+str(indexSF)+" -SF_grad TRUE -save_clouds"
    utils.Run(commande)

def poisson(filename,params):
    """Run Poisson Surface Reconstruction
    See docs http://www.cs.jhu.edu/~misha/Code/PoissonRecon/
    
    Args:
        filename (str): path ot input PLY file
        params (dict): parameter dictionary
            ex: {"bType":"Neumann","degree":"2",...}
    """
    query=utils.QUERY_0['PoissonRecon']+" --in "+filename+" --out "+filename[0:-4]+"_mesh.ply"
    for i in params.keys():
        query+=" --"+i+int(bool(len(params[i])))*" "
        if i in utils.POISSONRECON_PARAMETERS.keys():
            query+=utils.POISSONRECON_PARAMETERS[i][params[i]]
        else:
            query+=params[i]
    utils.Run(query)   

def rasterize(commande,grid_size,proj,empty):
    """
    Commande CC pour le calcul de grille
    """
    commande+=" -rasterize -grid_step "+str(grid_size)+" -vert_dir 2 -proj "\
               +proj+" -SF_proj "+proj
    if empty=="empty":
        commande+=" -output_cloud -save_clouds"
    else :
        commande+=" -empty_fill "+empty+\
                   " -output_cloud -save_clouds"
    utils.Run(commande)

def sample_mesh(query,density):
    query+=" -sample_mesh DENSITY "+str(density)+" -save_clouds"
    utils.Run_bis(query)

def seuillage(commande,indexSF,mini,maxi):
    """
    Commande CC pour le seuillage d'un Scalar Field
    """
    commande+=" -set_active_sf "+str(indexSF)+" -filter_sf "\
               +str(mini)+" "+str(maxi)+" -save_clouds"
    utils.Run(commande)

def subsampling(commande,min_dist):
    commande+=" -SS SPATIAL "+str(min_dist)+" -save_clouds"
    utils.Run(commande)
