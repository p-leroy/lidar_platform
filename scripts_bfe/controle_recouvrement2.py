# coding: utf-8
# Baptiste Feldmann

import glob, logging, os, pickle

import numpy as np
from joblib import Parallel,delayed

import plateforme_lidar as pl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Overlap(object):
    def __init__(self, workspace, m3c2_file, water_surface="", settings=[]):

        self.workspace = workspace
        self.m3c2_file = m3c2_file
        self.water_surface = water_surface

        self.cc_options = settings[0]
        self.root_names_dict = settings[1]
        self.max_uncertainty = settings[2]
        self.line_nb_digits = settings[3]

        self.keys = list(self.root_names_dict.keys())
        self.root_length=len(self.root_names_dict[self.keys[0]][0])
        self._preprocessingStatus=False

        self.folder = ""
        self.file_list = []
        self.listCompareLines = []
        self.overlapping_pairs = {}

    def _set_overlapping_pairs(self, pattern="*_thin.laz"):
        self.overlapping_pairs = []
        overlapping_pairs_pkl = os.path.join(self.workspace, "overlapping_pairs.pkl")
        if os.path.exists(overlapping_pairs_pkl):
            self.overlapping_pairs = pickle.load(open(overlapping_pairs_pkl, 'rb'))
        else:
            lines = os.path.join(self.workspace, pattern)  # only consider thin lines to investigate overlaps
            logger.info(f'self.root_length {self.root_length}, self.line_nb_digits {self.line_nb_digits}')
            self.overlapping_pairs, overlaps = pl.calculs.select_pairs_overlap(lines, [self.root_length, self.line_nb_digits])
            pickle.dump(self.overlapping_pairs, open(overlapping_pairs_pkl, 'wb'))
            overlaps = os.path.join(self.workspace, "overlaps.pkl")
            pickle.dump(overlaps, open(overlaps, 'wb'))

    def _get_name_from_line_number(self, line_number):
        test = True
        compt = 0
        while test:
            if int(line_number) <= self.keys[compt]:
                test = False
            else:
                compt += 1
        rootname = self.root_names_dict[self.keys[compt]]
        return rootname[0] + line_number + rootname[1]

    def _filtering(self, workspace, inFile, out_file, c2c=50, c2c_z=0.2):
        data = tools.lastools.read(workspace + inFile, extra_field=True)
        select = np.logical_or(data["c2c_absolute_distances"] > c2c, data["c2c_absolute_distances_z"] > c2c_z)
        out_data = tools.lastools.filter_las(data, select)
        out = os.path.join(workspace, out_file)
        tools.lastools.WriteLAS(out, out_data)
        return out

    def preprocessing(self, folder):
        print("[Overlap] Pre-processing...")

        self.folder = folder
        self.file_list = []
        self.listCompareLines = []

        dir_ = os.path.join(self.workspace, self.folder)

        if self.folder == "C2":
            self._set_overlapping_pairs()
            if self.overlapping_pairs is not {}:
                for num_a in self.overlapping_pairs.keys():
                    file_a = self._get_name_from_line_number(num_a)
                    file_core_pts = file_a[0:-4] + "_thin.laz"
                    for num_b in self.overlapping_pairs[num_a]:
                        file_b = self._get_name_from_line_number(num_b)
                        file_result = file_core_pts[0:-4] + "_m3c2_" + num_a + "and" + num_b + ".laz"
                        self.file_list += [[file_a, file_b, file_core_pts, file_result]]
                        self.listCompareLines += [num_a + "_" + num_b]
            else:
                raise ValueError("Overlapping pairs dictionary is empty")

        elif self.folder == "C3":
            tools.cloudcompare.c2c_files(self.cc_options,
                                         dir_,
                                         [os.path.split(i)[1] for i in glob.glob(os.path.join(dir_, "*_C3_r_thin.laz"))],
                                         os.path.join(self.workspace, self.water_surface),
                                         10,
                                         10)

            Parallel(n_jobs=20, verbose=1)(delayed(self._filtering)(dir_, i, i[0:-8] + "_1.laz")
                                          for i in [os.path.split(i)[1]
                                                    for i in glob.glob(os.path.join(dir_, "*_C3_r_thin_C2C.laz"))])

            for i in glob.glob(os.path.join(dir_, "*_C2C.laz")):
                os.remove(i)

            self._set_overlapping_pairs(motif="*_thin_1.laz")
            if len(self.pairsDict.keys()) == 0:
                raise ValueError("Comparison dictionary is empty")
            
            for i in self.pairsDict.keys():
                file_a = self._get_name_from_line_number(i)
                file_core_pts =  file_a[0:-4] + "_thin_1.laz"
                for num_b in self.pairsDict[i]:
                    file_b = self._get_name_from_line_number(num_b)
                    file_result = file_core_pts[0:-4] + "_m3c2_" + i + "and" + num_b + ".laz"
                    self.file_list += [[file_a, file_b, file_core_pts, file_result]]
                    self.listCompareLines += [i + "_"+num_b]

        elif self.folder == "C2_C3":
            tools.cloudcompare.c2c_files(self.cc_options,
                                         dir_,
                                         [os.path.split(i)[1] for i in glob.glob(os.path.join(dir_, "*_C2_r_thin.laz"))],
                                         os.path.join(self.workspace, self.water_surface),
                                         10,
                                         10)

            Parallel(n_jobs=20, verbose=1)(delayed(self._filtering)(dir_, i, i[0:-8] + "_1.laz")
                                          for i in [os.path.split(i)[1]
                                                    for i in glob.glob(os.path.join(dir_, "*_C2_r_thin_C2C.laz"))])
            for i in glob.glob(os.path.join(dir_, "*_C2C.laz")):
                os.remove(i)
            
            for file_a in [os.path.split(i)[1] for i in glob.glob(os.path.join(dir_, "*_C2_r.laz"))]:
                file_core_pts = file_a[0:-4] + "_thin_1.laz"
                file_b = file_a[0:-9] + "_C3_r.laz"
                file_result = file_core_pts[0:-4] + "_m3c2_C2C3.laz"
                self.file_list += [[file_a, file_b, file_core_pts, file_result]]
                self.listCompareLines += [i[self.root_length:self.root_length + self.line_nb_digits]]

        else:
            raise OSError("Unknown folder")
        
        self._preprocessingStatus = True
        print("[Overlap] Preprocessing done !")

    def processing(self):
        dir_ = os.path.join(self.workspace, self.folder)
        if self._preprocessingStatus:
            for i in range(0, len(self.file_list)):
                print("Measure distances with M3C2: " + self.listCompareLines[i])
                self.measure_distances_with_m3c2(*self.file_list[i])
                    
            print("[Overlap.processing] M3C2 analyzing...")
            self.results = Parallel(n_jobs=25, verbose=1)(
                delayed(self.filter_data)(
                    os.path.join(dir_, self.file_list[i][-1]), self.listCompareLines[i])
                for i in range(0, len(self.file_list))
            )
            np.savetxt(os.path.join(dir_, "save_results.txt"), self.results,
                       fmt='%s', delimiter=';', header='Comparaison;moyenne (m);ecart-type (m)')
            print("[Overlap.processing] M3C2 analyzing done !")
        else:
            raise OSError("[Overlap.processing] Preprocessing not done!")

    def measure_distances_with_m3c2(self, line_a, line_b, core_pts, out):
        path_a = os.path.join(self.workspace, line_a)
        path_b = os.path.join(self.workspace, line_b)
        path_core_pts = os.path.join(self.workspace, core_pts)
        query = tools.cloudcompare.open_file(self.cc_options, [path_a, path_b, path_core_pts])
        m3c2_params = os.path.join(self.workspace, self.m3c2File)
        tools.cloudcompare.m3c2(query, m3c2_params)
        root, ext = os.path.splitext(path_a)
        m3c2_expected_out = root + '_M3C2.laz'
        head, tail = os.path.split(path_a)
        dst = os.path.join(head, out)
        try:
            os.rename(m3c2_expected_out, dst)
            print(f'{m3c2_expected_out} renamed to {out}')
        except OSError as error:
            print(error)

    def filter_data(self, filepath, compare_id):
        in_data = tools.lastools.read(filepath, extra_field=True)
        selection = ~(np.isnan(in_data["distance__uncertainty"]))
        in_data2 = tools.lastools.filter_las(in_data, selection)
        selection = in_data2["distance__uncertainty"] < self.max_uncertainty
        in_data3 = tools.lastools.filter_las(in_data2, selection)
        selection = ~(np.isnan(in_data3['m3c2__distance']))
        m3c2_dist = in_data3['m3c2__distance'][selection]

        if len(m3c2_dist) > 100:
            output = [compare_id, np.round(np.mean(m3c2_dist), 3), np.round(np.std(m3c2_dist), 3)]
        else:
            output = [compare_id, "NotEnoughPoints", "-"]
        return output
