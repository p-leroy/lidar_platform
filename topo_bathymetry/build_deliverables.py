# coding: utf-8
# Paul Leroy
# Baptiste Feldmann
# formerly know as createDeliverables.py

import glob
import os
import shutil

from joblib import delayed, Parallel

import lidar_platform as lp
from topo_bathymetry import hole_filling as hf


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
    
    def __init__(self, workspace, pixel_size, root_name):

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

    def dtm(self, channel, tile_size, buffer, poisson_reconstruction=None, keep_only_16_in_c3=False):
        self.raster_dir = "_".join(["MNT", channel, self.pixel_size_name])
        self.raster_path = os.path.join(self.workspace, self.raster_dir)
        os.mkdir(self.raster_path)
        out_name = [self.root_name, "MNT", self.pixel_size_name + ".tif"]

        # copy Poisson reconstruction data
        if "PoissonRecon" in channel:
            print('[dtm] Copy Poisson reconstruction data')
            try:
                shutil.copy(poisson_reconstruction, self.raster_path)
            except FileNotFoundError:
                raise
        # extract ground related classes 2=ground (10=rail) 16=bathymetry
        if "C2" in channel:
            print('[dtm] Extract class 2 (ground) from C2 data')
            i = os.path.join(self.workspace, "LAS", "C2", "*.laz")
            lp.utils.run(f"las2las -i {i} -keep_class 2 -cores 50 -odir {self.raster_path} -olaz")

        if "C3" in channel:
            print('[dtm] Extract classes 2 (ground), 10 (rail) and 16 (bathymetry) from C3 data')
            i = os.path.join(self.workspace, "LAS", "C3", "*.laz")
            if keep_only_16_in_c3:
                lp.utils.run(f"las2las -i {i} -keep_class 16 -cores 50 -odir {self.raster_path} -olaz")
            else:
                lp.utils.run(f"las2las -i {i} -keep_class 2 10 16 -cores 50 -odir {self.raster_path} -olaz")

        if "FWF" in channel:
            print('[dtm] Extract class 16 (bathymetry) from FWF data')
            i = os.path.join(self.workspace, "LAS", "FWF", "*.laz")
            lp.utils.run(f"las2las -i {i} -keep_class 16 -cores 50 -odir {self.raster_path} -olaz")

        # build tiles from lines
        tiles_d = os.path.join(self.raster_path, "tiles")
        os.mkdir(tiles_d)
        lines = os.path.join(self.raster_path, "*.laz")
        o = os.path.join(self.raster_path, "tiles", self.root_name + "_MNT.laz")
        print('[dtm] Create spatial indexing information using lasindex')
        lp.utils.run(f"lasindex -i {lines} -cores 50")
        print('[dtm] Create tiles')
        lp.utils.run(f"lastile -i {lines} -tile_size {tile_size} -buffer {buffer} -cores 45 -odir {tiles_d} -o {o}")

        # blast to dem
        tiles = os.path.join(tiles_d, "*.laz")
        lp.utils.run(f"blast2dem -i {tiles} -step {self.pixel_size} -kill 250 -use_tile_bb -epsg 2154 -cores 50 -otif")

        # merge tif files
        i_tif = glob.glob(os.path.join(tiles_d, "*.tif"))
        o_merge = os.path.join(self.raster_path, "tiles", "_".join(out_name))
        lp.gdal.merge(i_tif, o_merge)

        clean(self.raster_path)

        return o_merge

    def dsm(self, channel, opt="vegetation"):  # opt for MNS : "vegetation" or "vegetation_building"
        self.raster_dir = "_".join(["MNS", opt, channel, self.pixel_size_name])
        self.raster_path = os.path.join(self.workspace, self.raster_dir)
        os.mkdir(self.workspace + self.raster_dir)
        out_name = [self.root_name, "MNS", self.pixel_size_name + ".tif"]
        odir = os.path.join(self.workspace + self.raster_dir)

        if "C2" in channel:
            i = os.path.join(self.workspace, 'LAS', 'C2', '*.laz')
            if opt == "vegetation":
                lp.utils.run(f"las2las -i {i} -keep_class 2 5 -cores 50 -odir {odir} -olaz")
            else:
                lp.utils.run(f"las2las -i {i} -keep_class 2 5 6 -cores 50 -odir {odir} -olaz")
        if "C3" in channel:
            i = os.path.join(self.workspace, 'LAS', 'C3', '*.laz')
            if opt == "vegetation":
                lp.utils.run(f"las2las -i {i} -keep_class 2 5 10 16 -cores 50 -odir {odir} -olaz")
            else:
                lp.utils.run(f"las2las -i {i} -keep_class 2 5 6 10 16 -cores 50 -odir {odir} -olaz")

        tiles_d = os.pathjoin(self.workspace, self.raster_dir, "tiles")
        thin_d = os.pathjoin(tiles_d, 'thin')
        os.mkdir(tiles_d)
        os.mkdir(thin_d)

        i = os.path.join(self.workspace, self.raster_dir,  "/*.laz")
        lp.utils.run(f"lasindex -i {i} -cores 50")
        odir = os.path.join(self.workspace, self.raster_dir, 'tiles')
        lp.utils.run(f"lastile -i {i} -tile_size 1000 -buffer 250 -cores 45 -odir {odir} -o {self.root_name}_MNS.laz")

        i = os.path.join(self.workspace, self.raster_dir, 'tiles', "/*.laz")
        odir = os.path.dir(self.workspace, self.raster_dir, "tiles", "thin")
        lp.utils.run(f"lasthin -i {i} -step 0.2 -highest -cores 50 -odir {odir} -olaz")

        i = os.path.join(self.workspace, self.raster_dir, "tiles", "thin", "*.laz")
        lp.utils.run(f"blast2dem -i {i} -step {self.pixel_size} -kill 250 -use_tile_bb -epsg 2154 -cores 50 -otif")

        tif = glob.glob(os.path.join(self.workspace, self.raster_dir, "tiles", "thin", "*.tif"))
        out = os.path.join(self.workspace, self.raster_dir, "tiles", "thin" + "_".join(out_name))
        lp.gdal.merge(tif, out)

        clean(self.raster_path)

        return out

    def dcm(self, channel):
        self.raster_dir = "_".join(["MNC", channel, self.pixel_size_name])
        self.raster_path = os.path.join(self.workspace, self.raster_dir)

        os.mkdir(self.raster_path)
        mnc_path = self.raster_path
        mns_path = os.path.join(self.workspace, "MNS_vegetation_" + channel + "_" + self.pixel_size_name, "thin")
        mnt_path = os.path.join(self.workspace,  "MNT_" + channel + "_" + self.pixel_size_name)

        if not (os.path.exists(mns_path + self.root_name + "_MNS_" + self.pixel_size_name + ".tif")
                and os.path.exists(mnt_path + self.root_name + "_MNT_" + self.pixel_size_name + ".tif")):
            raise OSError("MNS_vegetation or MNT aren't already computed !")

        out_name = [self.root_name, "MNC", self.pixel_size_name + ".tif"]
        list_mns = [os.path.split(i)[1] for i in glob.glob(mns_path + "*00.tif")]
        list_mnt = []
        list_mnc = []
        for i in list_mns:
            split_coords = i.split(sep="_")[-2::]
            list_mnt += [self.root_name + "_MNT_" + "_".join(split_coords)]
            list_mnc += [self.root_name + "_MNC_" + "_".join(split_coords)]
        Parallel(n_jobs=50, verbose=2)(
            delayed(pl.gdal.raster_calc)("A-B", mnc_path + list_mnc[i], mns_path + list_mns[i], mnt_path + list_mnt[i])
            for i in range(0, len(list_mns)))
        lp.gdal.merge(glob.glob(mnc_path + "*.tif"), mnc_path + "_".join(out_name))

    def density(self, channel, alt=True):
        dtm_path = os.path.join(self.workspace, "_".join(["MNT", channel, self.pixel_size_name]))
        if not os.path.exists(os.path.join(dtm_path, self.root_name + "_MNT_" + self.pixel_size_name + ".tif")):
            raise OSError("MNT isn't already computed !")

        out_name = [self.root_name, "MNT", "density", self.pixel_size_name + ".tif"]
        os.mkdir(os.path.join(dtm_path, "density"))  # FileExistsError
        os.mkdir(os.path.join(dtm_path, "density", "final"))

        lp.utils.run("lasgrid -i " + os.path.join(dtm_path, "*.laz") + f" -step {self.pixel_size}"
                     + " -use_tile_bb -counter_16bit -drop_class 10 -cores 50 -epsg 2154 -odir "
                     + os.path.join(dtm_path, 'density')
                     + " -odix _density -otif")
        list_mnt = [os.path.split(i)[1] for i in glob.glob(os.path.join(dtm_path, "*00.tif"))]

        # the parallel mode with joblib should be tested again, seems to fail sometimes
        for mnt in list_mnt:
            if alt:
                out = os.path.join(dtm_path, "density", "final", mnt[0:-4] + "_density.tif")
                hf.fill_holes(os.path.join(dtm_path, "density", mnt[0:-4] + "_density.tif"),
                              os.path.join(dtm_path, mnt),
                              out,
                              no_data_value=-9999
                              )
            else:
                lp.gdal.hole_filling(
                    os.path.join(dtm_path, "density", mnt[0:-4] + "_density.tif"),
                    os.path.join(dtm_path, mnt))

        lp.gdal.merge(
            glob.glob(os.path.join(dtm_path, "density", "final", "*.tif")),
            os.path.join(dtm_path, "density", "final", "_".join(out_name))
        )

        [os.remove(i) for i in glob.glob(os.path.join(dtm_path, "density", "*_density.tif"))]
        list_dir = os.listdir(os.path.join(dtm_path, "density", "final"))
        [shutil.move(os.path.join(dtm_path, "density", "final", i),
                     os.path.join(dtm_path, "density", i))
         for i in list_dir]

        shutil.rmtree(os.path.join(dtm_path, "density", "final"))

    def mkp(self, channel, mode):
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
        lp.utils.run(f"lasindex -i {i} -cores 50")
        lp.utils.run(f"lastile -i {i} -tile_size 1000 -buffer 25 -drop_class 0 -cores 45 -odir {tiles_d} -o MKP.laz")
        lp.utils.run(f"lasthin -i {tiles} + thinning -cores 50 -odix _thin -olaz")
        lp.utils.run(f"lastile -i {tiles_thin} -remove_buffer -cores 50 -olaz")
        lp.utils.run(f"lasmerge -i {tiles_thin_1} -odir {self.workspace} -o {out_name}")
        shutil.rmtree(self.raster_path)
