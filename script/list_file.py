# coding: utf-8
# Baptiste Feldmann
import glob
import os
import argparse

parser=argparse.ArgumentParser(description='Process some strings...')
parser.add_argument('-i', metavar='N', type=str)
args=parser.parse_args()
chemin=args.i
#chemin="G:/RENNES1/New_Zealand_Sept2019/05-Traitements/MNT/test/*.tif"

workspace=os.path.split(chemin)[0]+"/"
print(workspace)

liste=glob.glob(chemin)
liste_noms=[os.path.split(i)[1] for i in liste]
print(liste_noms)

try:
    f=open(workspace+"list_infiles.txt","w")
    for i in liste_noms:
        f.write(i+"\n")
    f.close()
except:
    raise OSError



