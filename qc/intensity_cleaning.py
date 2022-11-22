import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

print(f"__package__   {__package__}   +++++   __name__   {__name__}")

from ..tools import cc
from ..tools import misc
from ..config.config import cc_std

i_intensity = 0
i_intensity_in_raster = 1  # after the Height grid values field
i_imax_minus_i = 8

# correct the problem encountered in the Brioude data, C3 channel

def add_sf_imax_minus_i(line):
    print('[add_sf_imax_minus_i]')
    if not misc.exists(line):
        return

    head, tail, root, ext = misc.head_tail_root_ext(line)
    odir = os.path.join(head, 'lines_i_correction')
    if not os.path.exists(odir):
        os.makedirs(odir)
    pc_with_imax_minus_i = os.path.join(odir, root + '_i.sbf')
    raster = os.path.join(odir, root + '_raster.sbf')

    cmd = misc.cc_std
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    cmd += f' -O {line}'  # open the line
    cmd += f' -SET_ACTIVE_SF {i_intensity} -FILTER_SF MIN 400.'  # filter by intensity values
    cmd += ' -RASTERIZE -GRID_STEP 1 -VERT_DIR 2 -PROJ AVG -SF_PROJ MAX -OUTPUT_CLOUD'  # compute raster
    # rename the scalar field Intensity to a more convenient name for its later use: imax_minus_i
    # note: the raster is the only open entity at this time of the process
    cmd += f' -RENAME_SF {i_intensity_in_raster} imax_minus_i'
    cmd += f' -O {line}'  # re-open the line as it has been replaced in memory during -RASTERIZE -OUTPUT_CLOUD
    cmd += f' -SF_INTERP {i_intensity_in_raster}'  # interpolate scalar field from the raster to the original cloud
    cmd += f' -SF_OP_SF LAST SUB {i_intensity}'  # compute (imax - intensity), done in place
    cmd += f' -SAVE_CLOUDS FILE "{raster} {pc_with_imax_minus_i}"'

    misc.run(cmd)

    print(f'remove {raster}')
    os.remove(raster)
    os.remove(raster + '.data')

    return pc_with_imax_minus_i


def qc_check(pc_with_imax_minus_i, threshold=83, shift=103):
    print('[qc_check]')
    print(f'open {pc_with_imax_minus_i}')
    head, tail = os.path.split(pc_with_imax_minus_i)
    root, ext = os.path.splitext(tail)
    odir = os.path.join(head, 'qc')
    if not os.path.exists(odir):
        os.makedirs(odir)

    pc, sf, config = cc.read_sbf(pc_with_imax_minus_i)  # read the SBF file

    # limit intensity values range
    loc = sf[:, i_intensity] < 400
    sf = sf[loc, :]
    imax_minus_i = sf[:, i_imax_minus_i]
    intensity = sf[:, i_intensity]

    # low and high histograms
    print('compute and save histogram of corrected intensities')
    low = intensity[(threshold < imax_minus_i) & (imax_minus_i < 200)]
    high = intensity[(0 < imax_minus_i) & (imax_minus_i < threshold)]  - shift
    hist_low = np.histogram(low, bins=256)
    hist_high = np.histogram(high, bins=256)
    plt.plot(hist_low[1][:-1], hist_low[0], '.b', label='low intensities (not shifted)')
    plt.plot(hist_high[1][:-1], hist_high[0], '.r', label=f'high intensities shifted by {shift}')
    name = os.path.join(odir, root + '_histo_intensity_corrected')
    plt.xlabel('intensity bins')
    plt.ylabel('N')
    plt.xlim(0, 400)
    plt.title(tail)
    plt.legend()
    plt.grid()
    plt.savefig(name)
    plt.close()
    print(f'   => {name}')

    # intensity histogram
    print(f'compute and save histogram of original intensity')
    histogram = np.histogram(intensity, bins=256)
    i_corr = np.r_[low, high]
    histogram_corr = np.histogram(i_corr, bins=256)
    plt.plot(histogram_corr[1][:-1], histogram_corr[0] / 2, 'o', label='corrected intensity (N/2)')
    plt.plot(histogram_corr[1][:-1], histogram_corr[0], '.-', label='corrected intensity')
    plt.plot(histogram[1][:-1], histogram[0], '.-', label='initial intensity')
    name = os.path.join(odir, root + '_histo_intensity')
    plt.xlabel('intensity bins')
    plt.ylabel('N')
    plt.xlim(0, 400)
    plt.title(tail)
    plt.legend()
    plt.grid()
    plt.savefig(name)
    plt.close()
    print(f'   => {name}')


def correct_intensities_and_add_class(pc_with_imax_minus_i, threshold=83, shift=103):
    print('[correct_intensities_and_add_class]')
    if not os.path.exists(pc_with_imax_minus_i):
        print(f'file does not exists! {pc_with_imax_minus_i}')
        return

    head, tail = os.path.split(pc_with_imax_minus_i)
    root, ext = os.path.splitext(tail)
    high = os.path.join(head, root + '_high.sbf')
    low = os.path.join(head, root + '_low.sbf')
    merged = os.path.join(head, root + '_corr.sbf')

    cmd = cc_std
    # high intensities
    cmd += ' -SILENT -NO_TIMESTAMP -C_EXPORT_FMT SBF -AUTO_SAVE OFF'
    cmd += f' -O -GLOBAL_SHIFT FIRST {pc_with_imax_minus_i}'
    cmd += f' -SET_ACTIVE_SF LAST -FILTER_SF -10 {threshold}'  # filter (Imax - intensity)
    cmd += f' -SF_OP {i_intensity} SUB {shift}'  # shift intensity values, done in place
    cmd += ' -SF_ADD_CONST intensity_class 1'
    cmd += f' -SAVE_CLOUDS FILE {high}'
    cmd += ' -CLEAR_CLOUDS'
    # low intensities
    cmd += f' -O -GLOBAL_SHIFT FIRST {pc_with_imax_minus_i}'
    cmd += f' -SET_ACTIVE_SF LAST -FILTER_SF {threshold} 400'  # filter (Imax - intensity)
    cmd += ' -SF_ADD_CONST intensity_class 0'
    cmd += f' -SAVE_CLOUDS FILE {low}'
    # merge the clouds
    cmd += f' -O {high} -MERGE_CLOUDS -SAVE_CLOUDS FILE {merged}'
    misc.run(cmd)

    # remove temporary files
    print(f'remove {high}')
    os.remove(high)
    os.remove(high + '.data')
    print(f'remove {low}')
    os.remove(low)
    os.remove(low + '.data')
    print(f'remove {pc_with_imax_minus_i}')
    os.remove(pc_with_imax_minus_i)
    os.remove(pc_with_imax_minus_i + '.data')

    return merged

# correct the problem encountered in the Brioude dataset, C2 channel

def correct_intensity(sbf, global_shift, idx_intensity, debug=True):
    root, ext = os.path.splitext(sbf)
    head, tail = os.path.split(sbf)
    odir = os.path.join(head, 'i_corr')
    os.makedirs(odir, exist_ok=True)

    pc, sf, config = cc.read_sbf(sbf)
    name_index = cc.get_name_index_dict(config)
    scan_direction_flag = sf[:, name_index['ScanDirectionFlag']]

    select_scan_direction_flag_0 = scan_direction_flag == 0
    select_scan_direction_flag_1 = scan_direction_flag == 1

    sbf_0 = root + '_0.sbf'
    sbf_1 = root + '_1.sbf'

    # split the initial cloud in two clouds
    cc.write_sbf(sbf_0, pc[select_scan_direction_flag_0, :], sf[select_scan_direction_flag_0, :], config)
    cc.write_sbf(sbf_1, pc[select_scan_direction_flag_1, :], sf[select_scan_direction_flag_1, :], config)

    # call CloudCompare to interpolate the intensities and merge
    merged = cc.sf_interp_and_merge(sbf_1, sbf_0, idx_intensity, global_shift, debug=debug)

    os.remove(sbf_0)
    os.remove(sbf_0 + '.data')
    os.remove(sbf_1)
    os.remove(sbf_1 + '.data')
    out = os.path.join(odir, tail)
    shutil.move(merged, out)
    shutil.move(merged + '.data', out + '.data')

    return out
