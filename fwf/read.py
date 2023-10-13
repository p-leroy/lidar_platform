import numpy as np

from ..tools.las_fmt import unpack_vlr_record_waveform_packet_descriptor


def get_wpds(las_data):
    wpds = {}
    vlrs = las_data.header.vlrs
    for vlr in vlrs:
        if 99 < vlr.record_id < 355:
            wpds[vlr.record_id - 99] = unpack_vlr_record_waveform_packet_descriptor(vlr)

    return wpds


def get_waveform(index, las_data, wdp_filename=None, offset=0, make_positive=False, las_filename=None):
    # get the Waveform Packet Descriptor
    wpds = get_wpds(las_data)
    wpd = wpds[las_data.wavepacket_index[index]]
    # get the number of samples
    number_of_samples = wpd['number_of_samples']
    bytes_per_sample = int(wpd['bits_per_sample'] / 8)

    if las_data.header.global_encoding.waveform_data_packets_internal:
        with open(las_filename, 'rb') as bf:
            bf.seek(int(las_data.header.start_of_waveform_data_packet_record +
                        las_data.byte_offset_to_waveform_data[index]))
            count = number_of_samples * bytes_per_sample
            b = bf.read(count)
            a = np.frombuffer(b, dtype=np.uint16, count=number_of_samples)
            time = np.arange(number_of_samples) * wpd['temporal_sample_spacing']
            waveform = a * wpd['digitizer_gain'] + wpd['digitizer_offset'] + offset
            if make_positive:
                waveform = np.abs(waveform)
            return time, waveform

    if las_data.header.global_encoding.waveform_data_packets_external:
        with open(wdp_filename, 'rb') as bf:
            bf.seek(las_data.byte_offset_to_waveform_data[index])
            count = number_of_samples * bytes_per_sample
            b = bf.read(count)
            a = np.frombuffer(b, dtype=np.uint16, count=number_of_samples)
            time = np.arange(number_of_samples) * wpd['temporal_sample_spacing']
            waveform = a * wpd['digitizer_gain'] + wpd['digitizer_offset'] + offset
            if make_positive:
                waveform = np.abs(waveform)
            return time, waveform

    return None
