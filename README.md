
# **Lidar_platform**

## Installation
### 1. Install lidar_platform
The better way to install the lidar_platform module is by cloning the repository from GitHub:

https://github.com/p-leroy/lidar_platform

After that, you have to add the directory containing lidar_platform to your PYTHONPATH. This step is dependent on the way you installed python. In many editors, you can set the PYTHONPATH without touching to the environment variables.

Once the path have been configured, the module or the tools can be imported in a classical manner, e.g.:

```python
import lidar_platform
from lidar_platform import cc, las
```

### 2. Install required modules
Depending on your python installation, there are several ways to install modules. Sometimes, preferred ways are specified on the webstites of the modules. So, do not hesitate to go and have a look at the installation recommandations, which can evolute with time.

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
- To use tools.cloudcompare and tools.cc, you will need CloudCompare<br>
If CloudCompare is not installed in the standard directory ('C:\Program Files\CloudCompare' on Windows), configure the path in lidar_platform.config.config.py<br>
See more : http://www.cloudcompare.org/
- topo_bathymetry.poisson_reconstruction makes calls to the Adaptive Multigrid Solvers tools, especially PoissonRecon.exe<br>
See more: https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version13.8/
- In case of failure during the gdal installation, it is possible to try to install it using OSGeo4W<br>You will have to add the root path to OSGEO4W (containing OSGeo4W.bat) to your environment variables.<br>
See more : https://trac.osgeo.org/osgeo4w/

---

## Basic Usage
### Read / Write LAS file

```python
from lidar_platform import las
workspace = "D:/yourDirectory/"
dataset = las.read(workspace + "inFile.laz")
intensity = dataset.intensity
numberOfPoints = len(intensity)
...
addFieldList = [{"name": "addField1", "type": "float32", "data": extraField1},
                {"name": "addField2", "type": "uint8", "data": extraField2}]
las.WriteLAS(workspace + "outFile.laz", dataset, extraFields=addFieldList)
```

### Read / View / Filter full waveform LAS file

```python
from lidar_platform import las
from lidar_platform.fwf import las_fwf
workspace = "D:/yourDirectory/"
dataset = las.read(workspace + "inFile_fwf.laz")
waveforms = las.read_wdp(workspace + "inFile_fwf.laz", dataset)
indexPoint = 99
las_fwf.viewerFWF(las.filter_las(dataset, indexPoint), waveforms[indexPoint])
listPoints = [12, 102, 30]
dataExtract = las.filter_las(dataset, listPoints)
waveExtract = las.filter_wdp(waveforms, listPoints)
las.update_byte_offset(dataExtract, waveExtract)
las.WriteLAS(workspace + "outFile_fwf.laz",
             dataExtract, format_id=4, waveforms=waveExtract)
```
