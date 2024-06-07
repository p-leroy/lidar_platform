# coding: utf-8
# Baptiste Feldmann

import argparse
import glob

parser = argparse.ArgumentParser(description='Process some strings...')
parser.add_argument('-dirpath', metavar='N', type=str)

args = parser.parse_args()
workspace = args.dirpath

listFiles = glob.glob(workspace+"\\*.tif")

f = open(workspace + "\\list_infiles.txt", "w")
for i in listFiles:
    f.write(i + "\n")
f.close()
