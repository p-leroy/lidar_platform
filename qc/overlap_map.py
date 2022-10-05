# formerly known as carteRecouvrement.py [Baptiste Feldmann]

import glob
import logging
import os
import pickle

from joblib import delayed, Parallel

from lidar_platform import las, misc
from lidar_platform.tools import cloudcompare
from lidar_platform.topo_bathymetry.refraction_correction_helper_functions import select_pairs_overlap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Overlap(object):
    def __init__(self, workspace, m3c2_file, out, settings=[]):

        self.workspace = workspace
        self.m3c2File = m3c2_file
        self.out_name = out

        self.cc_options = settings[0]
        self.root_names_dict = settings[1]
        self.max_uncertainty = settings[2]
        self.max_dist = settings[3]
        self.line_nb_digits = settings[4]

        self.keys = list(self.root_names_dict.keys())
        self.root_length = len(self.root_names_dict[self.keys[0]][0])
        self._preprocessing_status = False
        self.file_list = []
        self.pair_list = []
        self.overlapping_pairs = []

    # PREPROCESSING

    def _set_overlapping_pairs(self, pattern="*_thin.laz"):
        self.overlapping_pairs = []
        overlapping_pairs_pkl = os.path.join(self.workspace, "overlapping_pairs.pkl")
        if os.path.exists(overlapping_pairs_pkl):
            print("[_set_overlapping_pairs] overlapping_pairs.pkl file found, do not run select_pairs_overlap")
            self.overlapping_pairs = pickle.load(open(overlapping_pairs_pkl, 'rb'))
        else:
            print("[_set_overlapping_pairs] compute overlapping pairs")
            lines = os.path.join(self.workspace, pattern)  # only consider thin lines to investigate overlaps
            logger.info(f'self.root_length {self.root_length}, self.line_nb_digits {self.line_nb_digits}')
            self.overlapping_pairs, overlaps = select_pairs_overlap(lines, [self.root_length, self.line_nb_digits])
            pickle.dump(self.overlapping_pairs, open(overlapping_pairs_pkl, 'wb'))
            overlaps_pkl = os.path.join(self.workspace, "overlaps.pkl")
            pickle.dump(overlaps, open(overlaps_pkl, 'wb'))

    def _select_root(self, line_number):
        test = True
        compt = 0
        while test:
            if int(line_number) <= self.keys[compt]:
                test = False
            else:
                compt += 1
        root_name = self.root_names_dict[self.keys[compt]]
        return root_name[0] + line_number + root_name[1]

    def preprocessing(self, pattern="*_thin.laz"):
        print("[Overlap.preprocessing]")
        self.file_list = []
        self.pair_list = []
        self._set_overlapping_pairs(pattern=pattern)
        if self.overlapping_pairs == {}:
            raise ValueError("Comparison dictionary is empty")

        for key in self.overlapping_pairs.keys():
            file_a = self._select_root(key)
            file_core_pts = file_a[0:-4] + "_thin.laz"
            for c in self.overlapping_pairs[key]:
                file_b = self._select_root(c)
                file_result = file_core_pts[0:-4] + "_m3c2_" + key + "and" + c + ".laz"
                self.file_list += [[file_a, file_b, file_core_pts, file_result]]
                self.pair_list += [key + "_" + c]
        self._preprocessing_status = True
        print("[Overlap.preprocessing] self.file_list")
        print(*self.file_list, sep='\n')
        print("[Overlap.preprocessing] done")

        return self.file_list

    def processing(self):
        if self._preprocessing_status:
            # compute M3C2 distances between lines
            for i in range(0, len(self.file_list)):
                print("[Overlap.processing] compute M3C2 distances: " + self.pair_list[i])
                self.measure_distances_with_m3c2(*self.file_list[i])

            print("[Overlap.processing] filter M3C2 data")
            Parallel(n_jobs=20, verbose=1)(
                delayed(self.filter_m3c2_data)(
                    os.path.join(self.workspace, self.file_list[i][-1]))
                for i in range(0, len(self.file_list))
            )
            out = os.path.join(self.workspace, self.out_name)
            print("[Overlap.processing] merge M3C2 data")
            query = "lasmerge -i " + os.path.join(self.workspace, "*_clean.laz") + " -o " + out
            misc.run(query)
            print("[Overlap.processing] M3C2 analyzes done")
        else:
            raise OSError("[Overlap.processing] preprocessing not done!")
        return out

    def clean_temporary_files(self):
        print("[Overlap.clean_temporary_files] remove temporary files (*_thin.laz, *_clean.laz)")
        [os.remove(i) for i in glob.glob(os.path.join(self.workspace, "*_thin.laz"))]
        [os.remove(i) for i in glob.glob(os.path.join(self.workspace + "*_clean.laz"))]

    def measure_distances_with_m3c2(self, line_a, line_b, core_pts, out):
        path_a = os.path.join(self.workspace, line_a)
        path_b = os.path.join(self.workspace, line_b)
        path_core_pts = os.path.join(self.workspace, core_pts)

        query = cloudcompare.open_file(self.cc_options, [path_a, path_b, path_core_pts])
        m3c2_params = os.path.join(self.workspace, self.m3c2File)
        cloudcompare.m3c2(query, m3c2_params)
        root, ext = os.path.splitext(path_a)
        m3c2_expected_out = root + '_M3C2.laz'
        head, tail = os.path.split(path_a)
        dst = os.path.join(head, out)
        try:
            os.rename(m3c2_expected_out, dst)
            print(f'{m3c2_expected_out} renamed to {out}')
        except OSError as error:
            print(error)

    def filter_m3c2_data(self, filepath):
        data = las.read(filepath, extra_field=True)
        extra_fields = [key for key in data.metadata['extraField']]

        try:  # filter distance uncertainty
            selection = data["distance__uncertainty"] < self.max_uncertainty
        except KeyError:
            raise KeyError(f"distance__uncertainty is not in the extra_fields list: {extra_fields}")

        try:  # filter m3c2 distance
            selection &= (data["m3c2__distance"] < self.max_dist)
            selection &= (data["m3c2__distance"] > -self.max_dist)
        except KeyError:
            raise KeyError(f"m3c2__distance is not in the extra_fields list: {extra_fields}")

        # save filtered data => *_clean.laz
        selected_data = las.filter_las(data, selection)
        extra = [(("m3c2_distance", "float32"), selected_data["m3c2__distance"]),
                 (("distance_uncertainty", "float32"), selected_data["distance__uncertainty"])]
        las.WriteLAS(filepath[0:-4] + "_clean.laz", selected_data, extra_fields=extra)
