#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function


import numpy as np
from matplotlib import pyplot as plt

color_cycle = [(31, 119, 180),
               (255, 127, 14),
               (44, 160, 44),
               (214, 39, 40),
               (148, 103, 189),
               (140, 86, 75),
               (227, 119, 194),
               (127, 127, 127),
               (188, 189, 34),
               (23, 190, 207)]


def get_color():
    get_color.color_cycle += 1
    if get_color.color_pointer > color_cycle:
        get_color.color_pointer = 0
    return color_cycle[get_color.color_cycle]


get_color.color_cycle += 0
