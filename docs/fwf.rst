.. _fwf:

======================
LAS/LAZ with waveforms
======================

Read
====

Note that ``lidar_platform.las`` makes calls to ``laspy``, you have to install this third party library.

::

   from lidar_platform import las
   las_data = las.read('filename')
   point_index = 0  # the index of the point in your LAS file
   time, waveform = las_data.get_waveform(point_index)
