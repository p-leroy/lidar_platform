from PIL import Image
import piexif
from numpy import loadtxt
from joblib import Parallel,delayed

def to_DMS(val):
    result=(int(val),int(60*(val-int(val))),3600*(val-int(val))-60*int(60*(val-int(val))))
    return result

def tag_file(filename,coords):
    img=Image.open(filename)
    alt=coords[2]
    lat_DMS,long_DMS=to_DMS(coords[0]),to_DMS(coords[1])

    gps_info={0:(2,3,0,0),
              1:"N",2:((lat_DMS[0],1),(lat_DMS[1],1),(int(lat_DMS[2]*10000),10000)),
              3:"E",4:((long_DMS[0],1),(long_DMS[1],1),(int(long_DMS[2]*10000),10000)),
              5: 0,6:(int(alt*100),100)}

    exif_data={'GPS':gps_info}
    exif_bytes=piexif.dump(exif_data)

    img.save(filename[0:-4]+"_geo.jpg","JPEG",quality=100,exif=exif_bytes)
    img.close()
    

workspace="G:/RENNES1/Suede-Abisko/08-Orthophotos/20170830/"
fichier="survey4_Index_bis.txt"
tableau=loadtxt(workspace+fichier,dtype=str,delimiter=';',skiprows=1,usecols=[7,8,9,10])

Parallel(n_jobs=45,verbose=3)(delayed(tag_file)(workspace+i[0],[float(i[1]),float(i[2]),float(i[3])]) for i in tableau)
