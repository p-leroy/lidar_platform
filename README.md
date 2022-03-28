## How to collect a specific folder on Plateforme Lidar's Git :
```
git init
git remote add -f origin https://github.com/baptistefeldmann/plateforme_lidar
git config core.sparseCheckout true
echo folderName > .git/info/sparse-checkout
git pull origin branchName
```

## What's new ?
No more need for laspy and pylas locally, we are moving to laspy 2.0 !!!

## How to upgrade from version 0.2 to version 0.3 ?
Install the new version of "plateforme_lidar" and don't forget to remove laspy and pylas libraries and install the new version of laspy with the following command:
```
pip uninstall laspy
pip uninstall pylas
pip install laspy[lazrs]
```

