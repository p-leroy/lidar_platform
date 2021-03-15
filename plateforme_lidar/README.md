# **Plateforme_lidar**

## Installation :
### 1. Install plateforme_lidar module :
add plateforme_lidar directory path to PYTHONPATH

### 2. Install required modules :
> pip install -r .\plateforme_lidar\requirements.txt

### 3. Install laspy and pylas :
> python .\plateforme_lidar\laspy\setup.py install<br>
> python .\plateforme_lidar\pylas\setup.py install

### 4. Useful software :
- Make sure that you have downloaded CloudCompare to use plateforme_lidar.cloudcompare.py<br>
Add path to cloudcompare.exe in plateforme_lidar.utils.py (dictionnary QUERY_0)<br>
Add path to PoissonSurfaceReconstruction.exe to use it in plateforme_lidar.utils.py (dictionnary QUERY_0)<br>
See more : http://www.cloudcompare.org/
- Make sure that you have downloaded OSGEO4W to use plateforme_lidar.gdal.py<br>
Add OSGEO4W in your environment variable<br>
See more : https://trac.osgeo.org/osgeo4w/

---

## Basic Usage
### Reading / Writing LAS file :
- .las / .laz without waveform packet : use lastools.readLAS_laspy() or lastools.readLAS()
- .las / .laz with waveform packet : use lastools.readLAS_laspy() only

```python
>>> from plateforme_lidar import pl
>>> workspace="D:/yourDirectory/"
>>> dataset = pl.lastools.readLAS(workspace+"inFile.laz")
>>> intensity = dataset.intensity
>>> numberOfPoints=len(data)
...
>>> addFieldList=[(("addField1","float32"),extraField1),(("addField2","uint8"),extraField2)]
>>> pl.lastools.writeLAS(workspace+"outFile.laz",dataset,extraField=extraField)
```

Reading, viewing and filtering fwf LAS file :
```python
>>> from plateforme_lidar import pl
>>> workspace="D:/yourDirectory/"
>>> dataset = lastools.readLAS_laspy(workspace+"inFile_fwf.laz")
>>> waveforms = lastools.readWDP(workspace+"inFile_fwf.laz",dataset)
>>> indexPoint=99
>>> pl.lasfwf.viewerFWF(pl.lastools.Filter_LAS(dataset,indexPoint),waveforms[indexPoint])
>>> listPoints=[12,102,30]
>>> dataExtract=pl.lastools.Filter_LAS(dataset,listPoints)
>>> waveExtract=pl.lastools.Filter_WDP(waveforms,listPoints)
>>> pl.lastools.Update_ByteOffset(dataExtract,waveExtract)
>>> pl.lastools.writeLAS(workspace+"outFile_fwf.laz",
dataExtract,format_id=4,waveforms=waveExtract)
```




