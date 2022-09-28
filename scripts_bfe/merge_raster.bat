@echo OFF
set /p workspace="Enter path : "
set /p projectName="Enter output filename: "

python G:\RENNES1\BaptisteFeldmann\Python\package\plateforme_lidar\list_file.py -dirpath %workspace%
set query=gdal_merge -n -9999 -a_nodata -9999 -ot Float32 -of GTiff -o %workspace%\%projectName%.tif --optfile %workspace%\list_infiles.txt

@echo ON
OSGeo4W %query% && del %workspace%\list_infiles.txt && pause