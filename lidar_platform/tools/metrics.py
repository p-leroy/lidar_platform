# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:15:30 2021

@author: PaulLeroy
"""

import configparser
import os
import shutil


def move_results(results, odir, ref_line, other_line):
    odir = os.path.join(*odir)
    os.makedirs(odir, exist_ok=True)
    tail = f'{ref_line}_{other_line}.sbf'
    out = os.path.join(odir, tail)
    shutil.move(results, out)
    shutil.move(results + '.data', out + '.data')


def build_m3c2_txt(txt, normal_scale, search_scale, search_depth):
    config = configparser.ConfigParser()
    config.optionxform = str
    with open(txt) as f:
        config.read_file(f)
    config['General']['normal_scale'] = str(normal_scale)
    config['General']['search_scale'] = str(search_scale)
    config['General']['search_depth'] = str(search_depth)
    head, tail = os.path.split(txt)
    out_tag = f'{normal_scale}_{search_scale}_{search_depth}'
    new_txt = os.path.join(head, f'm3c2_{out_tag}.txt')
    with open(new_txt, 'w') as f:
        config.write(f)
    return new_txt, out_tag
