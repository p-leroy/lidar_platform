@echo OFF
set /p workspace="Enter C3 dirpath : "
set n_jobs=45
cd %workspace%

mkdir tiles500m

@echo ON
lasindex -i *.laz -cores %n_jobs%
lastile -i *.laz -tile_size 500 -buffer 25 -cores %n_jobs% -odir tiles500m -o C3.laz
lasground -i ./tiles500m/*.laz -step 6 -nature -extra_fine -cores %n_jobs% -odix _g -olaz
las2las -i ./tiles500m/*_g.laz -keep_class 2 -cores %n_jobs% -odix round -olaz
lasthin -i ./tiles500m/*_g.laz -keep_class 2 -step 1 -lowest -cores %n_jobs% -odix _thin -olaz
lastile -i ./tiles500m/*_g_thin.laz -remove_buffer -cores %n_jobs% -olaz
lasmerge -i ./tiles500m/*_g_thin_1.laz -o C3_ground_thin_1m.laz

@echo OFF
del *.lax
del .\tiles500m\*_g.laz
del .\tiles500m\*_g_thin.laz
del .\tiles500m\*_g_thin_1.laz
pause
