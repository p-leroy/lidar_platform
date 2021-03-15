@echo OFF
set /p workspace="Enter C2 dirpath : "
set n_jobs=45
cd %workspace%

mkdir tiles500m\ground
mkdir tiles500m\other

@echo ON
lasindex -i *.laz -cores %n_jobs%
lastile -i *.laz -tile_size 500 -buffer 25 -cores %n_jobs% -odir tiles500m -o C2.laz
lasground -i ./tiles500m/*.laz -step 6 -nature -extra_fine -cores %n_jobs% -compute_height -odix _g -olaz
lasclassify -i ./tiles500m/*_g.laz -cores %n_jobs% -odix c -olaz
las2las -i ./tiles500m/*_gc.laz -keep_class 2 -cores %n_jobs% -odir ./tiles500m/ground -odix _ground -olaz
las2las -i ./tiles500m/*_gc.laz -drop_class 2 -cores %n_jobs% -odir ./tiles500m/other -odix _other -olaz
lastile -i ./tiles500m/ground/*_ground.laz -remove_buffer -cores %n_jobs% -olaz
lastile -i ./tiles500m/other/*_other.laz -remove_buffer -cores %n_jobs% -olaz

lasthin -i ./tiles500m/ground/*_ground.laz -step 1 -lowest -cores %n_jobs% -odix _thin -olaz
lastile -i ./tiles500m/ground/*_ground_thin.laz -remove_buffer -cores %n_jobs% -olaz
lasmerge -i ./tiles500m/ground/*_ground_thin_1.laz -o C2_ground_thin_1m.laz

@echo OFF
del *.lax
del .\tiles500m\*_g.laz
del .\tiles500m\*_gc.laz
del .\tiles500m\*_gc_ground_thin*.laz
pause
