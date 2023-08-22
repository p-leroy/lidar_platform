# coding: utf-8
# Paul Leroy
# Baptiste Feldmann


import os
import shutil
import time

import laspy
from joblib import Parallel, delayed
import numpy as np

from lidar_platform import las, sbet
from .refraction_correction_helper_functions import correction_3d, correction_vect


def refraction_correction(filepath, sbet_obj, minimum_depth=-0.1, force_sbet_std_time=False):
    root, ext = os.path.splitext(filepath)
    out = os.path.join(root + "_ref_corr.laz")

    # open bathymetry file and filter data by depth
    in_data = las.read(filepath, extra_fields=True)
    select = in_data.depth < minimum_depth
    data_under_water = las.filter_las(in_data, select)
    data_above_water = las.filter_las(in_data, np.logical_not(select))

    # GPS time format handling
    my_gps_time = las.GPSTime(in_data['gps_time'])
    in_data_gps_time = in_data['gps_time']
    detected_gps_time_format = my_gps_time.gps_time_type.name
    las_header_gps_time_format = laspy.open(filepath).header.global_encoding.gps_time_type.name

    if las_header_gps_time_format != detected_gps_time_format:
        msg = f'[refraction_correction] WARNING detected GPS time format ({detected_gps_time_format}) '
        msg += f'different from las header.global_encoding.gps_time_type ({las_header_gps_time_format})'
        print(msg)
        print('[refraction_correction] try to convert gps time to week time')
        week_number, week_time = my_gps_time.adjusted_standard_2_week_time()
        print(f'[refraction_correction] week number = {week_number}')
        gps_time = week_time[select]
    else:
        msg = '[refraction_correction] detected GPS time format '
        msg += f'and las header GPS time format are the same ({detected_gps_time_format})'
        print(msg)
        gps_time = data_under_water.gps_time

    # compute new positions
    apparent_depth = data_under_water.depth
    # data_interp = lp.sbet.interpolate(sbet_obj[0], sbet_obj[1], gps_time)
    if force_sbet_std_time:
        week_number = las.get_week_number(gps_time, adjusted=True)
        sbet_obj.gps_time = las.GPSTime(sbet_obj.gps_time).week_time_2_adjusted_standard(week_number=week_number)

    data_interp = sbet_obj.interpolate(gps_time)
    coords_true, true_depth = correction_3d(data_under_water.XYZ, apparent_depth, data_interp[:, 0:3])
    
    # write results in las files
    depth_all = np.concatenate((np.round(true_depth, decimals=2), np.array([None] * len(data_above_water))))
    extra = [(("depth", "float32"), depth_all)]
    data_under_water.XYZ = coords_true
    data_corbathy = las.merge_las([data_under_water, data_above_water])
    las.WriteLAS(out, data_corbathy, poinnt_format=1, extra_fields=extra)

    return out


def refraction_correction_fwf(filepath, minimum_depth=-0.1, output_suffix = "_corbathy"):

    # open bathymetry file
    in_data = las.read(filepath, True)

    # correct the beam vector
    vect_app = np.vstack([in_data.x_t, in_data.y_t, in_data.z_t]).transpose()
    vect_true_all = correction_vect(vect_app)
    in_data.x_t, in_data.y_t, in_data.z_t = vect_true_all[:, 0], vect_true_all[:, 1], vect_true_all[:, 2]

    # select data under water
    select = in_data.depth < minimum_depth
    data_under_water = las.filter_las(in_data, select)
    data_above_water = las.filter_las(in_data, np.logical_not(select))
    vect_app_under_water = vect_app[select]
    del in_data

    # compute new positions
    apparent_depth = data_under_water.depth
    coords_true, depth_true = correction_3d(data_under_water.XYZ, apparent_depth, vectorApp=vect_app_under_water)

    depth_all = np.concatenate((np.round(depth_true, decimals=2), np.array([None] * len(data_above_water))))
    extra = [(("depth", "float32"), depth_all)]

    data_under_water.XYZ = coords_true
    data_corbathy = las.merge_las([data_under_water, data_above_water])

    # write results in laz a file
    out = os.path.splitext(filepath)[0] + output_suffix + ".laz"
    las.WriteLAS(out, data_corbathy, point_format=9, extra_fields=extra)

    # simply copy the wdp file
    src = os.path.splitext(filepath)[0] + ".wdp"
    dst = os.path.splitext(out)[0] + ".wdp"
    shutil.copyfile(src, dst)


def do_work(files, sbet_params, n_jobs, fwf=False, minimum_depth=-0.1, force_sbet_std_time=False):
    start = time.time()

    if fwf:
        print("[Refraction correction] full waveform mode")
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(refraction_correction_fwf)(f)
            for f in files)
    else:
        print("[Refraction correction] SBET data processing: start")
        sbet_obj = sbet.sbet_config(sbet_params)
        print("[Refraction correction] SBET data processing: done")
        print("[Refraction correction] normal mode")
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(refraction_correction)(file, sbet_obj, force_sbet_std_time=force_sbet_std_time)
            for file in files)

    stop = time.time()
    print("[Refraction correction] done in " + str(round(stop - start, 1)) + " sec")

    return results


if __name__ == '__main__':
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Refraction correction command')
    parser.add_argument('input', type=str, help='input file to be corrected')
    parser.add_argument('sbet', type=str, help='SBET trajectory file')
    parser.add_argument('-fwf', action='store_true', help='full waveform')
    parser.add_argument('-n_jobs', metavar='N', type=int, default=1, help='number of jobs (1 by default)')
    args = parser.parse_args()
    # define parameters
    do_work(args.input, args.sbet, args.n_jobs, args.fwf)
