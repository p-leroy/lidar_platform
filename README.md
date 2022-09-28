
# **Lidar_platform**

## Installation
### 1. Install plateforme_lidar module
add lidar_platform path to the PYTHONPATH

### 2. Install required modules
With miniconda (or anaconda but the first one is preferred)
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

### 3. Useful software
- Make sure that you have downloaded CloudCompare to use tools.cloudcompare.py<br>
Add path to cloudcompare.exe in lidar_plateform.utils.py (dictionnary QUERY_0)<br>
See more : http://www.cloudcompare.org/
- Download the Adaptive Multigrid Solvers tools, especially PoissonRecon.exe and add path to PoissonRecon.exe to use it in lidar_platform.utils.py (dictionary QUERY_0)<br>
See more: https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version13.8/
- In case of failure during the gdal installation, it is possible to try to install it using OSGeo4W<br>You will have to add the root path to OSGEO4W (containing OSGeo4W.bat) to your environment variables.<br>
See more : https://trac.osgeo.org/osgeo4w/

---

## Basic Usage
### Reading / Writing LAS file

```python
>> > import lidar_platform as lp
>> > workspace = "D:/yourDirectory/"
>> > dataset = pl.lastools.read(workspace + "inFile.laz")
>> > intensity = dataset.intensity
>> > numberOfPoints = len(data)
...
>> > addFieldList = [{"name": "addField1", "type": "float32", "data": extraField1},
                     {"name": "addField2", "type": "uint8", "data": extraField2}]
>> > pl.lastools.WriteLAS(workspace + "outFile.laz", dataset, extraFields=addFieldList)
```

### Reading, viewing and filtering fwf LAS file

```python
>> > from plateforme_lidar import pl
>> > workspace = "D:/yourDirectory/"
>> > dataset = lastools.read(workspace + "inFile_fwf.laz")
>> > waveforms = lastools.read_wdp(workspace + "inFile_fwf.laz", dataset)
>> > indexPoint = 99
>> > pl.lasfwf.viewerFWF(pl.lastools.filter_las(dataset, indexPoint), waveforms[indexPoint])
>> > listPoints = [12, 102, 30]
>> > dataExtract = pl.lastools.filter_las(dataset, listPoints)
>> > waveExtract = pl.lastools.filter_wdp(waveforms, listPoints)
>> > pl.lastools.update_byte_offset(dataExtract, waveExtract)
>> > pl.lastools.WriteLAS(workspace + "outFile_fwf.laz",
                          dataExtract, format_id=4, waveforms=waveExtract)
```
