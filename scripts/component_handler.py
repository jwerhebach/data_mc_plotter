#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import glob

from plot_funcs import get_color


class Component:
    def __init__(self,
                 ctype,
                 label,
                 directory=None,
                 file_list=None,
                 max_files=-1,
                 n_files=1,
                 color=None,
                 livetime=1.,
                 aggregation=None,
                 keep_components=False,
                 weight=None):
        self.ctype = ctype
        self.label = label
        self.n_files = n_files
        self.livetime = livetime
        self.aggregation = aggregation
        self.keep_components = keep_components
        self.weight = weight
        if self.aggregation is None:
            assert (directory is not None) or (file_list is not None), \
                'If component is not from aggregation directory or file_list'\
                'is needed!'
            if file_list is None:
                self.file_list = glob.glob(directory)
            self.file_list = file_list
            if max_files > 0:
                upto = max(len(self.file_list), max_files)
                self.file_list[:upto]
        else:
            self.file_list = None

        if color is None:
            self.color = get_color()
