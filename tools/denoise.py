# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:07:15 2022

@author: PaulLeroy
"""

from . import misc


def lasnoise(i, odir, cores, step=4, isolated=5):
    misc.run(
        f'lasnoise -v -i {i} -remove_noise -step {step} -isolated {isolated} -odir {odir} -odix _denoised -olaz -cores {cores}')
