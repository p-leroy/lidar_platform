import glob
import os

from joblib import delayed, Parallel
import laspy
import numpy as np

import misc

class ReverseTiling(object):

    def __init__(self, workspace, rootname, buffer=False, cores=50, pt_src_id_as_line_number=True, id_name=None):
        # out a written in workspace/line
        # id_name dictionary: if specified, used to find the name of the output line from the point_source_id

        print("[Reverse Tiling memory friendly]")
        self.workspace = workspace
        self.cores = cores
        self.motif = rootname.split(sep='XX')
        self.pt_src_id_as_line_number = pt_src_id_as_line_number

        if id_name is None:
            self.id_name = {}
        else:
            self.id_name = id_name

        self.lines_dict = {}

        if buffer:
            print("[ReverseTiling] Remove buffer")
            self.remove_buffer()
        else:
            print('[ReverseTiling] no need to remove buffer')

        print("[ReverseTiling] Get a list of tiles for each point source id")
        self.get_point_source_ids_in_tiles()

        print("[ReverseTiling] Write flight lines")
        self.write_lines(buffer)

    def _get_pt_src_id_list(self, filename):
        f = laspy.read(filename)
        pt_src_id_array = np.unique(f.point_source_id)
        head, tail = os.path.split(filename)
        print(f'[_get_pt_src_id] {tail} {pt_src_id_array}')
        return tail, pt_src_id_array

    def _merge_lines(self, src_id, max_len, line_num=0):
        query = "lasmerge -i "
        for filename in self.lines_dict[src_id]:
            query += os.path.join(self.workspace,  filename) + " "

        if line_num == 0:
            diff = max_len - len(str(src_id))
            num_line = str(src_id)[0:-2]
        else:
            num_line = str(line_num)
            diff = max_len - len(num_line)

        if line_num != -1:
            name = self.motif[0] + "0" * diff + num_line + self.motif[1]
        else:
            name = self.id_name[src_id]
        odir = os.path.join(self.workspace, 'lines')
        os.makedirs(odir, exist_ok=True)
        o = os.path.join(odir, name)
        query += "-keep_point_source " + str(src_id) + " -o " + o
        misc.run(query)
        print(f'id {src_id}, name {o}')

    def remove_buffer(self):
        i = os.path.join(self.workspace, '*.laz')
        odir = os.path.join(self.workspace, "without_buffer")
        os.mkdir(self.workspace)
        query = f"lastile -i {i} -remove_buffer -cores {self.cores} -odir {odir} -olaz"
        misc.run(query)
        self.workspace = odir

    def get_point_source_ids_in_tiles(self):
        tiles = glob.glob(os.path.join(self.workspace, "*.laz"))
        result = Parallel(n_jobs=self.cores, verbose=0)(
            delayed(self._get_pt_src_id_list)(tile)
            for tile in tiles)
        self.lines_dict = {}
        for name, point_source_id_array in result:
            for point_source_id in point_source_id_array:
                if point_source_id not in self.lines_dict.keys():
                    self.lines_dict[point_source_id] = [name]
                else:
                    self.lines_dict[point_source_id].append(name)

    def write_lines(self, buffer):
        max_pt_src_id = len(str(max(self.lines_dict.keys())))
        max_number_lines = len(str(len(self.lines_dict.keys()) + 1))
        if self.pt_src_id_as_line_number:
            for i in self.lines_dict.keys():
                self._merge_lines(i, max_pt_src_id)
        else:
            num = 1
            list_pt_src_id = list(self.lines_dict.keys())
            list_pt_src_id.sort()
            for src_id in list_pt_src_id:
                print(f'[ReverseTiling.write_lines] point source id {src_id}')
                if self.id_name:
                    self._merge_lines(src_id, max_number_lines, -1)
                else:
                    self._merge_lines(src_id, max_number_lines, num)
                num += 1

        if buffer:
            names = [os.path.split(i)[1] for i in glob.glob(self.workspace + self.motif[0])]
            for name in names:
                os.rename(self.workspace + name, self.workspace[0:-9] + name)
            for filepath in glob.glob(os.path.join(self.workspace, '*_1.laz')):
                os.remove(filepath)
            os.rmdir(self.workspace)
