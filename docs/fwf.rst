.. _fwf:

======================
LAS/LAZ with waveforms
======================

Read
====

LAS (and its compressed counterpart LAZ), is a popular format for LiDAR point cloud and full waveform.

Note that ``lidar_platform.las`` makes calls to ``laspy``, you have to install this third party library first. The class ``lidar_platform.tootls.las.LasData`` inherits from ``laspy.LasData``.

::

   from lidar_platform import las
   las_data = las.read('filename')
   point_index = 0  # the index of the point in your LAS file
   time, waveform = las_data.get_waveform(point_index)

.. autoclass:: lidar_platform.tools.las.LasData
    :members:
    :undoc-members:
    :show-inheritance:
