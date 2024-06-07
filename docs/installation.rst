.. _installation:

============
Installation
============

lidar_platform
==============

The best way to install ``lidar_platform`` is by cloning the repository from GitHub:

`<https://github.com/p-leroy/lidar_platform>`_

After that, you have to add the directory containing ``lidar_platform`` to your PYTHONPATH. This step is dependent on the way you installed python. In many editors, you can set the PYTHONPATH without touching to the environment variables.

Once the path have been configured, the module or the tools can be imported in a classical manner, e.g.:

::

    import lidar_platform
    from lidar_platform import cc, las

Optional Python dependencies
============================

Depending on your python installation, there are several ways to install modules. Sometimes, preferred ways are
specified on the websites of the modules. So, do not hesitate to go and have a look at the installation
recommendations, which can evolve with time.

For instance, with miniconda (or anaconda but the first one is preferred)

.. code-block:: shell

    conda install -c conda-forge laspy

`<https://laspy.readthedocs.io/en/latest/installation.html>`_

.. code-block:: shell

    conda install -c conda-forge gdal

`<https://gdal.org/download.html>`_

.. code-block:: shell

    conda install -c conda-forge numpy, matplotlib, sklearn

You can also use pip, depending on your Python IDE.

.. code-block:: shell

    pip install -r .\plateforme_lidar\requirements.txt

Third party tools
=================

Paths to third party tools can be configured in lidar_platform.config.config.py.

Depending on what you need in the library, you will need to install third party tools:

* To use ``lidar_platform.tools.cloudcompare`` and ``lidar_platform.cc``, you will need CloudCompare. **If CloudCompare is not installed in the standard directory ('C:\Program Files\CloudCompare' on Windows), configure
  the path in lidar_platform.config.config.py.** See more : `<http://www.cloudcompare.org>`_
* ``topo_bathymetry.poisson_reconstruction`` makes calls to the Adaptive Multigrid Solvers tools, especially ```PoissonRecon.exe``` See more: https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version13.8/
* In case of failure during the ``gdal`` installation, it is possible to try to install it using OSGeo4W. You will have
  to add the root path to OSGEO4W (containing OSGeo4W.bat) to your environment variables. See more : `<https://trac.osgeo.org/osgeo4w>`_