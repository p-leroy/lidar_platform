# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:07:15 2022

@author: PaulLeroy
"""

from config import common_ple as ple


def lasnoise(i, odir, cores, step=4, isolated=5):
    ple.exe(
        f'lasnoise -v -i {i} -remove_noise -step {step} -isolated {isolated} -odir {odir} -odix _denoised -olaz -cores {cores}')
