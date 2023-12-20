# coding: utf-8
# Paul Leroy
# Baptiste Feldmann
# formerly know as createDeliverables.py

import glob
import os
import shutil

from joblib import delayed, Parallel

from lidar_platform import gdal, misc
from lidar_platform.topo_bathymetry import hole_filling as hf


def clean(path):

    print(f'clean {path}')

    laz = os.path.join(path, "*.laz")
    print(f'remove {laz}')
    [os.remove(i) for i in glob.glob(laz)]

    lax = os.path.join(path, "*.lax")
    print(f'remove {lax}')
    [os.remove(i) for i in glob.glob(lax)]

    tiles_d = os.path.join(path, 'tiles')
    files = glob.glob(os.path.join(tiles_d, "*"))
    print(f'move files from {tiles_d} to {path}')
    [shutil.move(file, path) for file in files]
    print(f'remove {tiles_d}')
    shutil.rmtree(tiles_d)


class Deliverable(object):
    
    def __init__(self, workspace, pixel_size, root_name,
                 input_c2=None, input_c3=None, input_fwf=None, poisson_path=None):

        self.workspace = workspace
        self.pixel_size = pixel_size
        self.root_name = root_name

        if self.pixel_size < 1:
            self.pixel_size_name = str(int(self.pixel_size * 100)) + "cm"
        else:
            self.pixel_size_name = str(int(self.pixel_size)) + "m"

        self.channel_settings = {"C2": ["C2"], "C3": ["C3"], "C2C3": ["C2", "C3"]}
        self.mkp_settings = {"ground": ["bathy", "ground"], "nonground": ["vegetation", "building"]}

        self.raster_path = None
        self.raster_dir = None

        if input_c2 is not None:
            self.i_c2 = input_c2
        else: # for backward compatibility
            self.i_c2 = os.path.join(self.workspace, "LAS", "C2", "*.laz")

        if input_c3 is not None:
            self.i_c3 = input_c3
        else: # for backward compatibility
            self.i_c3 = os.path.join(self.workspace, "LAS", "C3", "*.laz")

        if input_fwf is not None:
            self.i_fwf = input_fwf
        else: # for backward compatibility
            self.i_fwf = os.path.join(self.workspace, "LAS", "FWF", "*.laz")

        self.poisson_path = poisson_path

    def DTM(self, channel, tile_size, buffer, poisson_reconstruction=None, keep_only_16_in_c3=False):
        """
        Digital Terrain Model generation
        :param channel: a string containing one or several elements among 'C2', 'C3', 'FWF', 'PoissonRecon'
        :param tile_size: the tile size used during the computation
        :param buffer: the buffer size used when building tiles
        :param poisson_reconstruction: a file containing the data form the Poisson Reconstruction
        :param keep_only_16_in_c3: keep only the bathymetry in C3 data [True/False]
        :return:
        """
        self.raster_dir = "_".join(["DTM", channel, self.pixel_size_name])
        self.raster_path = os.path.join(self.workspace, self.raster_dir)
        os.mkdir(self.raster_path)
        out_name = [self.root_name, "DTM", self.pixel_size_name + ".tif"]

        # copy Poisson reconstruction data
        if "PoissonRecon" in channel:
            print('[DTM] Copy Poisson reconstruction data')
            try:
                shutil.copy(poisson_reconstruction, self.raster_path)
            except FileNotFoundError:
                raise

        # extract ground related classes from C2 data [2=ground]
        if "C2" in channel:
            print(f'[DTM] Extract class 2 (ground) from C2 data ({self.i_c2})')
            misc.run(f"las2las -i {self.i_c2} -keep_class 2 -cores 50 -odir {self.raster_path} -olaz")

        # extract ground related classes from C3 data [2=ground, 10=rail, 16=bathymetry]
        if "C3" in channel:
            if keep_only_16_in_c3:
                print(f'[DTM] Extract classes 16 (bathymetry) from C3 data ({self.i_c3})')
                misc.run(f"las2las -i {self.i_c3} -keep_class 16 -cores 50 -odir {self.raster_path} -olaz")
            else:
                print(f'[DTM] Extract classes 2 (ground), 10 (rail) and 16 (bathymetry) from C3 data ({self.i_c3})')
                misc.run(f"las2las -i {self.i_c3} -keep_class 2 10 16 -cores 50 -odir {self.raster_path} -olaz")

        # extract ground related classes from C3 full waveform data [16=bathymetry]
        if "FWF" in channel:
            print(f'[DTM] Extract class 16 (bathymetry) from FWF data ({self.i_fwf})')
            misc.run(f"las2las -i {self.i_fwf} -keep_class 16 -cores 50 -odir {self.raster_path} -olaz")

        # build tiles from lines
        tiles_d = os.path.join(self.raster_path, "tiles")
        os.mkdir(tiles_d)
        lines = os.path.join(self.raster_path, "*.laz")
        o = os.path.join(self.raster_path, "tiles", self.root_name + "_DTM.laz")
        print('[DTM] Create spatial indexing information using lasindex')
        misc.run(f"lasindex -i {lines} -cores 50")
        print('[DTM] Create tiles')
        misc.run(f"lastile -i {lines} -tile_size {tile_size} -buffer {buffer} -cores 45 -odir {tiles_d} -o {o}")

        # blast to dem
        tiles = os.path.join(tiles_d, "*.laz")
        misc.run(f"blast2dem -i {tiles} -step {self.pixel_size} -kill 250 -use_tile_bb -epsg 2154 -cores 50 -otif")

        # merge tif files
        i_tif = glob.glob(os.path.join(tiles_d, "*.tif"))
        o_merge = os.path.join(self.raster_path, "tiles", "_".join(out_name))
        gdal.merge(i_tif, o_merge)

        clean(self.raster_path)

        return o_merge

    def DSM(self, channel, opt="vegetation"):  # opt for MNS : "vegetation" or "vegetation_building"
        self.raster_dir = "_".join(["DSM", opt, channel, self.pixel_size_name])
        self.raster_path = os.path.join(self.workspace, self.raster_dir)
        os.mkdir(self.workspace + self.raster_dir)
        out_name = [self.root_name, "DSM", self.pixel_size_name + ".tif"]
        odir = os.path.join(self.workspace + self.raster_dir)

        if "C2" in channel:
            if opt == "vegetation":
                misc.run(f"las2las -i {self.i_c2} -keep_class 2 5 -cores 50 -odir {odir} -olaz")
            else:
                misc.run(f"las2las -i {self.i_c2} -keep_class 2 5 6 -cores 50 -odir {odir} -olaz")

        if "C3" in channel:
            if opt == "vegetation":
                misc.run(f"las2las -i {self.i_c3} -keep_class 2 5 10 16 -cores 50 -odir {odir} -olaz")
            else:
                misc.run(f"las2las -i {self.i_c3} -keep_class 2 5 6 10 16 -cores 50 -odir {odir} -olaz")

        tiles_d = os.pathjoin(self.workspace, self.raster_dir, "tiles")
        thin_d = os.pathjoin(tiles_d, 'thin')
        os.mkdir(tiles_d)
        os.mkdir(thin_d)

        i = os.path.join(self.workspace, self.raster_dir,  "*.laz")
        misc.run(f"lasindex -i {i} -cores 50")
        odir = os.path.join(self.workspace, self.raster_dir, 'tiles')
        misc.run(f"lastile -i {i} -tile_size 1000 -buffer 250 -cores 45 -odir {odir} -o {self.root_name}_MNS.laz")

        i = os.path.join(self.workspace, self.raster_dir, 'tiles', "*.laz")
        odir = os.path.dir(self.workspace, self.raster_dir, "tiles", "thin")
        misc.run(f"lasthin -i {i} -step 0.2 -highest -cores 50 -odir {odir} -olaz")

        i = os.path.join(self.workspace, self.raster_dir, "tiles", "thin", "*.laz")
        misc.run(f"blast2dem -i {i} -step {self.pixel_size} -kill 250 -use_tile_bb -epsg 2154 -cores 50 -otif")

        tif = glob.glob(os.path.join(self.workspace, self.raster_dir, "tiles", "thin", "*.tif"))
        out = os.path.join(self.workspace, self.raster_dir, "tiles", "thin" + "_".join(out_name))
        gdal.merge(tif, out)

        clean(self.raster_path)

        return out

    def DCM(self, channel):
        self.raster_dir = "_".join(["DCM", channel, self.pixel_size_name])
        self.raster_path = os.path.join(self.workspace, self.raster_dir)

        os.mkdir(self.raster_path)
        mnc_path = self.raster_path
        mns_path = os.path.join(self.workspace, "DSM_vegetation_" + channel + "_" + self.pixel_size_name, "thin")
        dtm_path = os.path.join(self.workspace,  "DTM_" + channel + "_" + self.pixel_size_name)

        if not (os.path.exists(mns_path + self.root_name + "_DSM_" + self.pixel_size_name + ".tif")
                and os.path.exists(dtm_path + self.root_name + "_DTM_" + self.pixel_size_name + ".tif")):
            raise OSError("DSM_vegetation or DTM aren't already computed !")

        out_name = [self.root_name, "DCM", self.pixel_size_name + ".tif"]
        list_mns = [os.path.split(i)[1] for i in glob.glob(mns_path + "*00.tif")]
        list_dtm = []
        list_mnc = []
        for i in list_mns:
            split_coords = i.split(sep="_")[-2::]
            list_dtm += [self.root_name + "_DTM_" + "_".join(split_coords)]
            list_mnc += [self.root_name + "_DCM_" + "_".join(split_coords)]
        Parallel(n_jobs=50, verbose=2)(
            delayed(gdal.raster_calc)("A-B", mnc_path + list_mnc[i], mns_path + list_mns[i], dtm_path + list_dtm[i])
            for i in range(0, len(list_mns)))
        gdal.merge(glob.glob(mnc_path + "*.tif"), mnc_path + "_".join(out_name))

    def density(self, channel, alt=True):
        dtm_path = os.path.join(self.workspace, "_".join(["DTM", channel, self.pixel_size_name]))
        if not os.path.exists(os.path.join(dtm_path, self.root_name + "_DTM_" + self.pixel_size_name + ".tif")):
            raise OSError("DTM isn't already computed !")

        out_name = [self.root_name, "DTM", "density", self.pixel_size_name + ".tif"]
        os.mkdir(os.path.join(dtm_path, "density"))  # FileExistsError
        os.mkdir(os.path.join(dtm_path, "density", "final"))

        misc.run("lasgrid -i " + os.path.join(dtm_path, "*.laz") + f" -step {self.pixel_size}"
                     + " -use_tile_bb -counter_16bit -drop_class 10 -cores 50 -epsg 2154 -odir "
                     + os.path.join(dtm_path, 'density')
                     + " -odix _density -otif")
        list_dtm = [os.path.split(i)[1] for i in glob.glob(os.path.join(dtm_path, "*00.tif"))]

        # the parallel mode with joblib should be tested again, seems to fail sometimes
        for dtm in list_dtm:
            if alt:
                out = os.path.join(dtm_path, "density", "final", dtm[0:-4] + "_density.tif")
                hf.fill_holes(os.path.join(dtm_path, "density", dtm[0:-4] + "_density.tif"),
                              os.path.join(dtm_path, dtm),
                              out,
                              no_data_value=-9999
                              )
            else:
                gdal.hole_filling(
                    os.path.join(dtm_path, "density", dtm[0:-4] + "_density.tif"),
                    os.path.join(dtm_path, dtm))

        gdal.merge(
            glob.glob(os.path.join(dtm_path, "density", "final", "*.tif")),
            os.path.join(dtm_path, "density", "final", "_".join(out_name))
        )

        [os.remove(i) for i in glob.glob(os.path.join(dtm_path, "density", "*_density.tif"))]
        list_dir = os.listdir(os.path.join(dtm_path, "density", "final"))
        [shutil.move(os.path.join(dtm_path, "density", "final", i),
                     os.path.join(dtm_path, "density", i))
         for i in list_dir]

        shutil.rmtree(os.path.join(dtm_path, "density", "final"))

    def MKP(self, channel, mode):
        # mode : ground or nonground
        # settings : ground=[vertical, horiz], nonground=[thinning, step]

        self.raster_dir = "MKP_" + mode
        out_name = "_".join(self.root_name, "MKP", channel, mode + ".laz")
        os.mkdir(os.path.join(self.workspace, self.raster_dir))
        os.mkdir(os.path.join(self.workspace, self.raster_dir, "tiles"))

        if mode == "ground":
            folder_list = ["C3_bathy"] + [i + "ground" for i in self.channel_settings[channel]]
        elif mode == "nonground":
            folder_list = []
            for i in ["vegetation", "building"]:
                folder_list += [c + i for c in self.channel_settings[channel]]
        else:
            raise OSError("MKP works only for ground or nonground modes")

        for folder in folder_list:
            laz = glob.glob(os.path.join(self.workspace, "LAS", folder, "*.laz"))
            Parallel(n_jobs=50, verbose=1)(
                delayed(shutil.copy)(i, os.path.join(self.workspace, self.raster_dir, os.path.split(i)[1]))
                for i in laz)

        i = os.path.join(self.workspace, self.raster_dir, "*.laz")
        tiles_d = os.path.join(self.raster_path, "tiles")
        tiles = os.path.join(tiles_d, '*.laz')
        tiles_thin = os.path.join(tiles_d, '*_thin.laz')
        tiles_thin_1 = os.path.join(tiles_d, '*_thin_1.laz')
        misc.run(f"lasindex -i {i} -cores 50")
        misc.run(f"lastile -i {i} -tile_size 1000 -buffer 25 -drop_class 0 -cores 45 -odir {tiles_d} -o MKP.laz")
        misc.run(f"lasthin -i {tiles} + thinning -cores 50 -odix _thin -olaz")
        misc.run(f"lastile -i {tiles_thin} -remove_buffer -cores 50 -olaz")
        misc.run(f"lasmerge -i {tiles_thin_1} -odir {self.workspace} -o {out_name}")
        shutil.rmtree(self.raster_path)
