echo OFF
set workspace=G:/RENNES1/Moselle_Sept2018/05-Traitements_bathy/
echo ON
python correction_bathy_command.py -i %workspace%Moselle_20180919_C3_r_class_bathy_nocor.laz -sbet params_sbet_Eure.txt -n_jobs 1
pause