.. _sbf:

======================
SBF Simple Binary File
======================

A module to handle files in the SBF format (Simple Binary File). SBF is an internal format of CloudCompare used to store single point clouds with their associated scalar fields.

::

    from lidar_platform import sbf
    sbf_data = sbf.read(sbf_file)  # sbf_data is an object of type SbfData
    xyz = sbf_data.xyz  # coordinates of the points as a NumPy array
    scalar_fields = sbf_data.sf  # scalar fields as a NumPy array
    scalar_fields_names = sbf_data.sf_names  # scalar fields names

.. automodule:: lidar_platform.tools.sbf
    :imported-members:
    :members:
    :undoc-members:
    :show-inheritance:
