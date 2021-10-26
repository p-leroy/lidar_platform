# coding: utf-8
# Baptiste Feldmann
from plateforme_lidar import calculs
import argparse,time
import numpy as np
import importlib
importlib.reload(calculs)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Process some strings...')

    parser.add_argument('-dirpath', metavar='N', type=str)
    parser.add_argument('-root', metavar='N', type=str)
    parser.add_argument('-buffer',action='store_true',default=False)
    parser.add_argument('-cores',default=50)
    parser.add_argument('-o_ptsrcid',action='store_true',default=False)

    args=parser.parse_args()
    workspace=args.dirpath
    name=args.root
    calculs.ReverseTiling_mem(workspace,name,args.buffer,args.cores,args.o_ptsrcid)

