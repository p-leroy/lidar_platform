# coding: utf-8
#
# formerly known as controle_recouvrement2.py [Baptiste Feldmann]

import errno
import glob
import logging
import os
import pickle

import numpy as np
from joblib import Parallel, delayed

from lidar_platform  import las
from lidar_platform.tools import cloudcompare
from lidar_platform.topo_bathymetry.refraction_correction_helper_functions import select_pairs_overlap
from ..tools import cc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Overlap(object):
    def __init__(self, workspace, lines_dir_a, settings, m3c2_file, water_surface="", lines_dir_b=""):

        self.workspace = workspace
        self.m3c2_file = m3c2_file
        self.water_surface = os.path.join(workspace, water_surface)

        self.cc_options = settings[0]
        self.line_template_a = settings[1]
        self.max_uncertainty = settings[2]
        self.root_length = len(self.line_template_a[0])
        self.line_nb_digits = self.line_template_a[1]

        self._preprocessingStatus = False
        self.folder = ""
        self.odir = ""
        self.lines_dir_a = lines_dir_a
        self.lines_dir_b = lines_dir_b
        self.file_list = []
        self.pair_list = []
        self.overlapping_pairs = {}

    # PREPROCESSING

    def _set_overlapping_pairs(self, pattern="*_thin.laz"):
        self.overlapping_pairs = []
        overlapping_pairs_pkl = os.path.join(self.odir, "overlapping_pairs.pkl")
        if os.path.exists(overlapping_pairs_pkl):
            print("[_set_overlapping_pairs] overlapping_pairs.pkl file found, do not run select_pairs_overlap")
            self.overlapping_pairs = pickle.load(open(overlapping_pairs_pkl, 'rb'))
        else:
            print("[_set_overlapping_pairs] compute overlapping pairs")
            lines = os.path.join(self.odir, pattern)  # only consider thin lines to investigate overlaps
            logger.info(f'self.root_length {self.root_length}, line_nb_digits {self.line_nb_digits}')
            self.overlapping_pairs, overlaps = select_pairs_overlap(lines, [self.root_length, self.line_nb_digits])
            pickle.dump(self.overlapping_pairs, open(overlapping_pairs_pkl, 'wb'))
            overlaps_pkl = os.path.join(self.odir, "overlaps.pkl")
            pickle.dump(overlaps, open(overlaps_pkl, 'wb'))

    def reset_internals(self, folder):
        self.file_list = []
        self.pair_list = []
        self.folder = folder
        self.odir = os.path.join(self.workspace, self.folder)

    def build_lists(self, pattern, use_water_surface=False):
        self._set_overlapping_pairs(pattern=pattern)

        if self.overlapping_pairs is {}:
            raise ValueError("Overlapping pairs dictionary is empty")

        for num_a in self.overlapping_pairs.keys():
            name_a = self.line_template_a[0] + num_a + self.line_template_a[-1]
            file_a = os.path.join(self.lines_dir_a, name_a)
            head, tail = os.path.split(file_a)
            if use_water_surface:
                file_core_pts = os.path.join(self.odir, tail[0:-4] + "_thin_core.laz")
            else:
                file_core_pts = os.path.join(self.odir, tail[0:-4] + "_thin.laz")
            for num_b in self.overlapping_pairs[num_a]:
                name_b = self.line_template_a[0] + num_b + self.line_template_a[-1]
                file_b = os.path.join(self.lines_dir_a, name_b)
                file_result = file_core_pts[0:-4] + "_m3c2_" + num_a + "and" + num_b + ".sbf"
                self.file_list += [[file_a, file_b, file_core_pts, file_result]]
                self.pair_list += [num_a + "_" + num_b]

    def _filtering_c2c(self, in_file, out_file, c2c=50, c2c_z=0.2):
        head, tail = os.path.split(in_file)
        data = las.read(in_file, extra_field=True)

        try:  # C2C absolute distances
            select_c2c = data["c2c__absolute__distances"] > c2c
        except KeyError:
            raise KeyError(f"c2c__absolute__distances is not in the extra_fields list")
        try:  # C2C absolute distances Z
            select_c2c_z = data["c2c__absolute__distances__z"] > c2c_z
        except KeyError:
            raise KeyError(f"c2c__absolute__distances__z is not in the extra_fields list")
        select = select_c2c | select_c2c_z

        out_data = las.filter_las(data, select)
        out = os.path.join(head, out_file)
        las.WriteLAS(out, out_data)
        return out

    def select_points_away_from_water_surface(self, pattern):
        # compute distances between thin data and the water surface => *_thin_C2C.laz
        thin_files = glob.glob(os.path.join(self.odir, pattern))
        octree_lvl = 10
        nbr_job = 10
        cloudcompare.c2c_files(self.cc_options,
                                  thin_files,
                                  self.water_surface,
                                  octree_lvl=octree_lvl,
                                  nbr_job=nbr_job)

        # keep only points which are far from the water surface
        # *_thin_C2C.laz files are created at the end of the previous command
        thin_c2c_files = glob.glob(os.path.join(self.odir, "*_thin_C2C.laz"))
        Parallel(n_jobs=20, verbose=1)(
            delayed(self._filtering_c2c)(file, file[0:-8] + "_core.laz")
            for file in thin_c2c_files
        )

        for i in glob.glob(os.path.join(self.odir, "*_C2C.laz")):
            os.remove(i)

    def preprocessing(self, folder, pattern="*_thin.laz", use_water_surface=False):
        print(f"[Overlap.preprocessing] folder: {folder}")
        self.reset_internals(folder)
        self.build_lists(pattern, use_water_surface)

        if use_water_surface:
            # build *_thin_core.laz files from *_thin.laz and water surface
            self.select_points_away_from_water_surface(pattern)

        self._preprocessingStatus = True
        print(f"[Overlap.preprocessing] done (use_water_surface = {use_water_surface})")

        return self.file_list

    def preprocessing_c2_c3(self, folder, lines_dir_b, line_template_b, pattern="*_thin.laz", use_water_surface=False):
        print(f"[Overlap.preprocessing_c2_c3] folder: {folder}")
        self.reset_internals(folder)

        if use_water_surface:
            # build *_thin_core.laz files from *_thin.laz and water surface
            self.select_points_away_from_water_surface(pattern)

        if use_water_surface:
            core_files = glob.glob(os.path.join(self.odir, "*_thin_core.laz"))
        else:
            core_files = glob.glob(os.path.join(self.odir, "*_thin.laz"))
        for file_core in core_files:
            head, tail = os.path.split(file_core)
            num_a = tail[self.root_length: self.root_length + self.line_nb_digits]
            file_a = os.path.join(self.lines_dir_a, self.line_template_a[0] + num_a + self.line_template_a[-1])
            file_b = os.path.join(lines_dir_b, line_template_b[0] + num_a + line_template_b[-1])
            file_result = file_core[0:-4] + "_m3c2_C2C3.sbf"
            self.file_list += [[file_a, file_b, file_core, file_result]]
            head, tail = os.path.split(file_core)
            self.pair_list += [tail[self.root_length:self.root_length + self.line_nb_digits]]
        
        self._preprocessingStatus = True
        print("[Overlap.preprocessing_c2_c3] done")

        return self.file_list

    # PROCESSING

    def measure_distances_with_m3c2(self, line_a, line_b, core_pts, out):
        # do the files exist?
        path_a = os.path.join(self.odir, line_a)
        print(f'Cloud 1: {path_a}')
        if not os.path.exists(path_a):
            raise FileNotFoundError
        path_b = os.path.join(self.odir, line_b)
        print(f'Cloud 2: {path_b}')
        if not os.path.exists(path_b):
            raise FileNotFoundError
        path_core_pts = os.path.join(self.odir, core_pts)
        print(f'Core points: {path_core_pts}')
        if not os.path.exists(path_core_pts):
            raise FileNotFoundError
        m3c2_params = os.path.join(self.odir, self.m3c2_file)
        print(f'M3C2 parameters: {m3c2_params}')
        if not os.path.exists(m3c2_params):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), m3c2_params)

        # compute M3C2
        cc_options = [self.cc_options[0], 'SBF_auto_save', self.cc_options[-1]]
        query = cloudcompare.open_file(cc_options, [path_a, path_b, path_core_pts])
        cloudcompare.m3c2(query, m3c2_params)
        root, ext = os.path.splitext(path_a)
        expected_sbf = root + '_M3C2.sbf'
        head, tail = os.path.split(path_a)
        out_sbf = os.path.join(head, out)
        if os.path.exists(out_sbf):
            os.remove(out_sbf)
            os.remove(out_sbf + ".data")
            print(f'[Overlap.measure_distances_with_m3c2] remove {out_sbf}')
        try:
            os.rename(expected_sbf, out_sbf)
            os.rename(expected_sbf + '.data', out_sbf + '.data')
            print(f'{expected_sbf} (.sbf.data also) renamed to {out} ')
        except OSError as error:
            print(error)

    def filter_m3c2_data_sbf(self, filepath, compare_id):
        print(f'[Overlap.filter_m3c2_data_sbf] {filepath} [{compare_id}]')
        pc, sf, config = cc.read_sbf(filepath)

        name_index_dict = cc.get_name_index_dict(config)
        i_uncertainty = name_index_dict['distance uncertainty']
        i_distance = name_index_dict['M3C2 distance']
        uncertainty = sf[:, i_uncertainty]
        distance = sf[:, i_distance]

        # filter distance uncertainty
        selection = ~(np.isnan(uncertainty))
        selection &= (uncertainty < self.max_uncertainty)

        # filter m3c2 distance
        selection &= ~(np.isnan(distance))

        # compute statistics on the selected M3C2 distances (mean, standard deviation)
        m3c2_dist = distance[selection]
        if len(m3c2_dist) > 100:
            output = [compare_id, np.round(np.mean(m3c2_dist), 3), np.round(np.std(m3c2_dist), 3)]
        else:
            output = [compare_id, "NotEnoughPoints", "-"]

        # save filtered data => *_clean.sbf
        root, ext = os.path.splitext(filepath)
        out = root + '_clean.sbf'
        cc.write_sbf(out, pc[selection, :], sf[selection, :], config)

        return output

    def processing(self):
        if self._preprocessingStatus:
            for i in range(0, len(self.file_list)):
                print("[Overlap.processing] Measure distances between lines with M3C2: " + self.pair_list[i])
                self.measure_distances_with_m3c2(*self.file_list[i])
                    
            print("[Overlap.processing] filter M3C2 results and compute statistics (mean, standard deviation)")
            self.results = Parallel(n_jobs=10, verbose=1)(
                delayed(self.filter_m3c2_data_sbf)(
                    os.path.join(self.odir, elem[-1]), self.pair_list[count])
                for count, elem in enumerate(self.file_list)
            )
            np.savetxt(os.path.join(self.odir, "m3c2_mean_std.txt"), self.results,
                       fmt='%s', delimiter=';', header='Comparaison;moyenne (m);ecart-type (m)')

            cleaned_m3c2_results = glob.glob(os.path.join(self.odir, '*_clean.sbf'))
            # if there are several files, merge them
            if len(cleaned_m3c2_results) == 1:
                print(f"[Overlap.processing] only one output file: {cleaned_m3c2_results[0]}")
            elif len(cleaned_m3c2_results) == 0:
                print(f"[Overlap.processing] no output file, this is quite unexpected")
            else:
                overlap_control_src = cc.merge(cleaned_m3c2_results, export_fmt='sbf')
                overlap_control_dst = os.path.join(self.workspace, f'{self.folder}_overlap_control.sbf')
                os.rename(overlap_control_src, overlap_control_dst)
                os.rename(overlap_control_src + '.data', overlap_control_dst + '.data')
                print(f"[Overlap.processing] results merged in {overlap_control_dst}")

            print("[Overlap.processing] M3C2 analyzes done")
        else:
            raise OSError("[Overlap.processing] Preprocessing not done!")
