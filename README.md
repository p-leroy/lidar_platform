## How to collect a specific folder on Plateforme Lidar's Git :
```
git init
git remote add -f origin https://github.com/baptistefeldmann/plateforme_lidar
git config cor.sparseCheckout true
echo folderName > .git/info/sparse-checkout
git pull origin branchName
```
