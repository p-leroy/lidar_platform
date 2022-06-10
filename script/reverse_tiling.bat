@echo OFF
rem set /p rootname="Enter rootName : "
rem set /p workspace="Enter path : "
rem set /p test_buff="Buffer ? (0/1) : "
set rootname = Aude_20200214-02_LXX_C2_r_new.laz
set workspace = G:\RENNES1\Aude_fevrier2020\05-Traitements\C2\classification\BandingCorrectionDisabled
set test_buff = 0

echo ON
python G:/RENNES1/BaptisteFeldmann/Python/package/plateforme_lidar/reverse_tiling_use.py -dirpath %workspace:\=/%/ -root %rootname% -buffer %test_buff%
pause