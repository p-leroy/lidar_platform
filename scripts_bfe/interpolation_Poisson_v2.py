import plateforme_lidar as PL
import numpy as np
import glob,os,time

import importlib
importlib.reload(PL)

def interpolation(workspace,filename,surface_water,params):
    #params=['tile_size','bbox','CC','interpolation','normals']
    deb=time.time()
    print(filename)
    print("==============")
    # if not os.path.exists(workspace+filename[0:-4]+"_normals.ply"):
    #     PL.cloudcompare.compute_normals(workspace+filename,params['normals'])
    #     PL.cloudcompare.last_file(workspace+filename[0:-4]+"_20*.ply",filename[0:-4]+"_normals.ply")
    # PL.cloudcompare.poisson(workspace+filename[0:-4]+"_normals.ply",params['interpolation'])
    # PL.cloudcompare.sample_mesh(PL.cloudcompare.open_file(params['CC'],workspace+filename[0:-4]+"_normals_mesh.ply"),5)
    # PL.cloudcompare.last_file(workspace+filename[0:-4]+"_normals_mesh_SAMPLED_POINTS_20*.laz",filename[0:-4]+"_sample_mesh.laz")
    # os.remove(workspace+filename[0:-4]+"_normals.ply")
    # os.remove(workspace+filename[0:-4]+"_normals_mesh.ply")

    bbox=np.array(filename[0:-4].split(sep="_")[params['bbox']:params['bbox']+2]+[params['tile_size']],dtype=str)
    tools.cloudcompare.las2las_keep_tile(workspace + filename[0:-4] + "_sample_mesh.laz", bbox)
    
    query= tools.cloudcompare.open_file(params['CC'], [workspace + filename[0:-4] + "_sample_mesh_1.laz", workspace + surface_water])
    tools.cloudcompare.c2c_dist(query, True, 10)
    tools.cloudcompare.last_file(workspace + filename[0:-4] + "_sample_mesh_1_C2C_DIST_20*.laz", filename[0:-4] + "_sample_mesh_1_C2C.laz")
    temp= tools.cloudcompare.last_file(workspace + surface_water[0:-4] + "_20*.laz")
    os.remove(temp)
    data= tools.lastools.read(workspace + filename[0:-4] + "_sample_mesh_1_C2C.laz", extra_field=True)
    select=np.logical_and(data.c2c_absolute_distances<100,
                          np.logical_and(data.c2c_absolute_distances_z>-10,data.c2c_absolute_distances_z<-1))
    
    os.remove(workspace+filename[0:-4]+"_sample_mesh_1_C2C.laz")
    if any(select):
        data_new= tools.lastools.filter_las(data, select)
        tools.lastools.WriteLAS(workspace + filename[0:-4] + "_sample_mesh_1_1.laz", data_new)
    
        query= tools.cloudcompare.open_file(params['CC'], [workspace + filename[0:-4] + "_sample_mesh_1_1.laz", workspace + filename])
        tools.cloudcompare.c2c_dist(query, False, 10)
        tools.cloudcompare.last_file(workspace + filename[0:-4] + "_sample_mesh_1_1_C2C_DIST_20*.laz", filename[0:-4] + "_sample_mesh_1_1_C2C.laz")
        temp= tools.cloudcompare.last_file(workspace + filename[0:-4] + "_20*.laz")
        os.remove(temp)
        data= tools.lastools.read(workspace + filename[0:-4] + "_sample_mesh_1_1_C2C.laz", extra_field=True)
        select=np.logical_and(data.c2c_absolute_distances>0.5,
                              data.c2c_absolute_distances<200)
        os.remove(workspace+filename[0:-4]+"_sample_mesh_1_1_C2C.laz")
        if any(select):
            data_new= tools.lastools.filter_las(data, select)
            tools.lastools.WriteLAS(workspace + filename[0:-4] + "_sample_mesh_step1.laz", data_new)
        else:
            print("Pas de point")
        os.remove(workspace+filename[0:-4]+"_sample_mesh_1_1.laz")
    else:
        print("Pas de point")
    os.remove(workspace+filename[0:-4]+"_sample_mesh_1.laz")
    print("Duration : %.1f sec\n" %(time.time()-deb))

def interp_step2(workspace,filename,surface_water,params):
    #params=['window','CC','interpolation','normals']
    #window=[minX,minY,maxX,maxY]
    deb=time.time()
    print(filename)
    print("==============")
    tools.cloudcompare.compute_normals(workspace + filename, params['normals'])
    tools.cloudcompare.last_file(workspace + filename[0:-4] + "_20*.ply", filename[0:-4] + "_normals.ply")
    tools.cloudcompare.poisson(workspace + filename[0:-4] + "_normals.ply", params['interpolation'])
    tools.cloudcompare.sample_mesh(
        tools.cloudcompare.open_file(params['CC'], workspace + filename[0:-4] + "_normals_mesh.ply"), 5)
    tools.cloudcompare.last_file(workspace + filename[0:-4] + "_normals_mesh_SAMPLED_POINTS_20*.laz", filename[0:-4] + "_sample_mesh.laz")
    tools.cloudcompare.las2las_clip_xy(workspace + filename[0:-4] + "_sample_mesh.laz", params['window'])
    os.remove(workspace+filename[0:-4]+"_normals_mesh.ply")
    os.remove(workspace+filename[0:-4]+"_normals.ply")
    os.remove(workspace+filename[0:-4]+"_sample_mesh.laz")

    query= tools.cloudcompare.open_file(params['CC'], [workspace + filename[0:-4] + "_sample_mesh_1.laz", workspace + surface_water])
    tools.cloudcompare.c2c_dist(query, True, 10)
    tools.cloudcompare.last_file(workspace + filename[0:-4] + "_sample_mesh_1_C2C_DIST_20*.laz", filename[0:-4] + "_sample_mesh_1_C2C.laz")
    temp= tools.cloudcompare.last_file(workspace + surface_water[0:-4] + "_20*.laz")
    os.remove(temp)
    data= tools.lastools.read(workspace + filename[0:-4] + "_sample_mesh_1_C2C.laz", extra_field=True)
    select=np.logical_and(data.c2c_absolute_distances<100,
                          np.logical_and(data.c2c_absolute_distances_z>-10,data.c2c_absolute_distances_z<-1))
    os.remove(workspace+filename[0:-4]+"_sample_mesh_1_C2C.laz")
    if any(select):
        data_new= tools.lastools.filter_las(data, select)
        tools.lastools.WriteLAS(workspace + filename[0:-4] + "_sample_mesh_1_1.laz", data_new)
    
        query= tools.cloudcompare.open_file(params['CC'], [workspace + filename[0:-4] + "_sample_mesh_1_1.laz", workspace + filename])
        tools.cloudcompare.c2c_dist(query, False, 10)
        tools.cloudcompare.last_file(workspace + filename[0:-4] + "_sample_mesh_1_1_C2C_DIST_20*.laz", filename[0:-4] + "_sample_mesh_1_1_C2C.laz")
        temp= tools.cloudcompare.last_file(workspace + filename[0:-4] + "_20*.laz")
        os.remove(temp)
        data= tools.lastools.read(workspace + filename[0:-4] + "_sample_mesh_1_1_C2C.laz", extra_field=True)
        select=np.logical_and(data.c2c_absolute_distances>0.5,
                              data.c2c_absolute_distances<200)
        os.remove(workspace+filename[0:-4]+"_sample_mesh_1_1_C2C.laz")
        if any(select):
            data_new= tools.lastools.filter_las(data, select)
            tools.lastools.WriteLAS(workspace + filename[0:-4] + "_sample_mesh_final.laz", data_new)
        else:
            print("Pas de point")
        os.remove(workspace+filename[0:-4]+"_sample_mesh_1_1.laz")
    else:
        print("Pas de point")
    os.remove(workspace+filename[0:-4]+"_sample_mesh_1.laz")
    print("Duration : %.1f sec\n" %(time.time()-deb))

def neighbors_4(coords,tile_size):
    #return XY lower left coordinates of all neighbors in 4-connect
    #coords=[X,Y] lower left
    listCoords={"left":np.array([coords[0]-tile_size,coords[1]],dtype=int),
                "up":np.array([coords[0],coords[1]+tile_size],dtype=int),
                "right":np.array([coords[0]+tile_size,coords[1]],dtype=int),
                "down":np.array([coords[0],coords[1]-tile_size],dtype=int)}
    return listCoords

def listing_neighbors(listFilenames,params):
    #params=['bbox','tile_size']
    dictio={}
    for i in listFilenames:
        splitFilename=i[0:-4].split(sep="_")
        coords=np.array(splitFilename[params['bbox']:params['bbox']+2],dtype=int)
        listNeigh=neighbors_4(coords,params['tile_size'])
        dictio[i]=dict(zip(listNeigh.keys(),["","","",""]))
        for c in listNeigh.keys():
            filename="_".join(splitFilename[0:params['bbox']]+[str(listNeigh[c][0]),str(listNeigh[c][1])]+splitFilename[params['bbox']+2::])+".laz"
            if filename in listFilenames and filename not in dictio.keys():
                dictio[i][c]=filename
    return dictio

def bbox_edge(coords,params,buffer=0):
    #coords=[X,Y] lower left
    #params=["edge","tile_size","dist"]
    #dictio=[minX,minY,maxX,maxY]
    dictio={"right":[params['tile_size']-params["dist"],-buffer,params['tile_size']+params["dist"],params["tile_size"]+buffer],
            "left":[-params["dist"],-buffer,params["dist"],params["tile_size"]+buffer],
            "up":[-buffer,params['tile_size']-params["dist"],params['tile_size']+buffer,params["tile_size"]+params['dist']],
            "down":[-buffer,-params["dist"],params['tile_size']+buffer,params['dist']]}
    bbox=np.array([coords[0],coords[1],coords[0],coords[1]])
    bbox+=np.array(dictio[params['edge']])
    return np.array(bbox,dtype=int)  


workspace=r'G:\RENNES1\Loire_totale_automne2019\Loire_Briare-Sully-sur-Loire\05-Traitements\C3\interpolation'+'\\'

liste_noms=[os.path.split(i)[1] for i in glob.glob(workspace+"*00.laz")]
surf_water="C2_ground_thin_1m_watersurface_smooth5.laz"

bbox_place=1 #start at 0, separator="_"
tile_size=2000
dist_buffer=250
dist_cut=50

params_CC=['standard','LAS','Loire45-4']

params_normal={"shiftname":"Loire45-4",
               "normal_radius":"2",
               "model":"QUADRIC"}

params_interp={"bType":"Neumann",
               "degree":"2",
               "width":"4",
               "scale":"2",
               "samplesPerNode":"5",
               "pointWeight":"100",
               "threads":"45",
               "density":"",
               "performance":"",
               "verbose":""}


#----STEP 1----#
#Compute Poisson interpolation for all tiles
for i in liste_noms:
    interpolation(workspace,i,surf_water,{"interpolation":params_interp,"CC":params_CC,"normals":params_normal,"bbox":bbox_place,"tile_size":tile_size})

#For each tile, cut edges when there is neighbors tiles (in 4-connect)
for i in liste_noms:
    print(i)
    if os.path.exists(workspace+i[0:-4]+"_sample_mesh_step1.laz"):
        data= tools.lastools.read(workspace + i[0:-4] + "_sample_mesh_step1.laz")
        splitname=i[0:-4].split(sep="_")
        coords_LL=[int(splitname[bbox_place]),int(splitname[bbox_place+1])]
        dictioNeigh=neighbors_4(coords_LL,tile_size)
        select=np.array([True]*len(data))
        for c in dictioNeigh.keys():
            filename="_".join(splitname[0:bbox_place]+[str(dictioNeigh[c][0]),str(dictioNeigh[c][1])]+splitname[bbox_place+2::])
            if os.path.exists(workspace+filename+"_sample_mesh_step1.laz"):
                bbox_cut=bbox_edge(coords_LL,{"edge":c,"tile_size":tile_size,"dist":dist_cut})
                selectX=np.logical_and(data.XYZ[:,0]>bbox_cut[0],data.XYZ[:,0]<bbox_cut[2])
                selectY=np.logical_and(data.XYZ[:,1]>bbox_cut[1],data.XYZ[:,1]<bbox_cut[3])
                select=np.logical_and(select,np.logical_not(np.logical_and(selectX,selectY)))
        if any(select):
            tools.lastools.WriteLAS(workspace + i[0:-4] + "_sample_mesh_step1cut.laz", tools.lastools.filter_las(data, select))
#=======================#

#---STEP 2-----#
#Compute Poisson interpolation for all overlap area
listNeighbors=listing_neighbors(liste_noms,{"bbox":bbox_place,"tile_size":tile_size})
for i in listNeighbors.keys():
    print(i)
    if os.path.exists(workspace+i[0:-4]+"_sample_mesh_step1cut.laz"):
        splitname=i[0:-4].split(sep="_")
        coords_LL=np.array([splitname[bbox_place],splitname[bbox_place+1]],dtype=int)
        for c in listNeighbors[i]:
            if os.path.exists(workspace+listNeighbors[i][c][0:-4]+"_sample_mesh_step1cut.laz"):
                query="lasmerge -i "+workspace+i+" "+workspace+i[0:-4]+"_sample_mesh_step1cut.laz "+\
                       workspace+listNeighbors[i][c][0:-4]+"_sample_mesh_step1cut.laz -o "+workspace+i[0:-4]+"_step2_"+c+".laz"
                utils.run(query)
                bbox_cut=np.array(bbox_edge(coords_LL,{"edge":c,"tile_size":tile_size,"dist":dist_buffer},dist_buffer),dtype=str)
                query="las2las -i "+workspace+i[0:-4]+"_step2_"+c+".laz -keep_xy "+" ".join(bbox_cut)+" -odix cut -olaz"
                utils.run(query)
                bbox_cut=np.array(bbox_edge(coords_LL,{"edge":c,"tile_size":tile_size,"dist":dist_cut}),dtype=str)
                interp_step2(workspace,i[0:-4]+"_step2_"+c+"cut.laz",surf_water,{"interpolation":params_interp,"CC":params_CC,"normals":params_normal,"window":bbox_cut})
#==============================#

#Merge interpolation files rom Step1 and those from Step2 (overlap area)
if not os.path.exists(workspace+"merge"):
    os.mkdir(workspace+"merge")
utils.run("lasmerge -i " + workspace + "*_sample_mesh_step1cut.laz -o " + workspace + "merge/interpolation_step1cut.laz")
utils.run("lasmerge -i " + workspace + "*_sample_mesh_final.laz -o " + workspace + "merge/interpolation_step2final.laz")
    


