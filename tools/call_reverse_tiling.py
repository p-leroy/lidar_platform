# coding: utf-8
# Baptiste Feldmann

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some strings...')

    parser.add_argument('-dirpath', required=True, type=str, help="directory path with slash at the end")
    parser.add_argument('-root', required=True, type=str, help="rootname for flightlines with XX for locating line num")
    parser.add_argument('-buffer', type=int, choices=[0,1], default=0, help="1 if buffer, 0 if not")
    parser.add_argument('-cores', type=int, choices=range(1,os.cpu_count()), default=50, help="number of cpu used")
    parser.add_argument('-o_ptsrcid', type=int, choices=[0,1], default=0, help="1 if you want ptsrcid as line number, 0 if not")

    args = parser.parse_args()
    workspace = args.dirpath
    name = args.root
    calculs.ReverseTiling(workspace,
                          name,
                          bool(args.buffer),
                          args.cores,
                          bool(args.o_ptsrcid))
