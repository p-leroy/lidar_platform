
# **Lidar_platform**

## Installation

### 1. Install lidar_platform

The best way to install the ```lidar_platform``` module is by cloning the repository from GitHub:

https://github.com/p-leroy/lidar_platform

After that, you have to add the directory containing lidar_platform to your PYTHONPATH. This step is dependent on the way you installed python. In many editors, you can set the PYTHONPATH without touching to the environment variables.

Once the path have been configured, the module or the tools can be imported in a classical manner, e.g.:

```python
import lidar_platform
from lidar_platform import cc, las
```

### 2. Install required modules
Depending on your python installation, there are several ways to install modules. Sometimes, preferred ways are 
specified on the websites of the modules. So, do not hesitate to go and have a look at the installation 
recommendations, which can evolve with time.

For instance, with miniconda (or anaconda but the first one is preferred)
> conda install -c conda-forge laspy 
> 
> https://laspy.readthedocs.io/en/latest/installation.html
> 
> conda install -c conda-forge gdal
> 
> https://gdal.org/download.html#
> 
> conda install -c conda-forge scikit-learn
> 
> And also numpy, matplotlib, sklearn

or with pip

> pip install -r .\plateforme_lidar\requirements.txt

### 3. Third party tools

Paths to third party tools can be configured in lidar_platform.config.config.py.

Depending on what you need in the library, you will need to install third party tools:
- To use ```lidar_platform.tools.cloudcompare``` and ```lidar_platform.cc```, you will need CloudCompare.
**If CloudCompare is not installed in the standard directory ('C:\Program Files\CloudCompare' on Windows), configure 
  the path in lidar_platform.config.config.py.** See more : http://www.cloudcompare.org
- ```topo_bathymetry.poisson_reconstruction``` makes calls to the Adaptive Multigrid Solvers tools, especially ```PoissonRecon.exe``` See more: https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version13.8/
- In case of failure during the `gdal` installation, it is possible to try to install it using OSGeo4W. You will have 
  to add the root path to OSGEO4W (containing OSGeo4W.bat) to your environment variables.
See more : https://trac.osgeo.org/osgeo4w

# HOWTO

## Read a LAS/LAZ with waveforms

Note that ````lidar_platform.las```` makes calls to ```laspy```, you have to install this third party library.

```python
from lidar_platform import las
las_data = las.read('filename')
point_index = 0  # the index of the point in your LAS file
time, waveform = las_data.get_waveform(point_index)
```

## Build deliverables

We have helper functions to generate deliverables as DTM, DSM, DCM, which make calls to `lastools blast2dem` (not free 
unfortunately) and to 
`gdal` (thank you OSGeo!). Under Windows, `gdal` can be installed with OsGeo4W (https://trac.osgeo.org/osgeo4w). Once 
installed, add `C:\QGIS` or `C:\OSGeo4W` to your path, it depends on your configuration.
