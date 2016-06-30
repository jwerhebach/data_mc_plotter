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

for i in range(len(color_cycle)):
    r, g, b = color_cycle[i]
    color_cycle[i] = (r / 255., g / 255., b / 255)

lw = 2.



def get_color():
    get_color.color_pointer += 1
    if get_color.color_pointer >= len(color_cycle):
        get_color.color_pointer = 0
    return color_cycle[get_color.color_pointer]

get_color.color_pointer = 0


def plot_hist(ax, label, hist, binning, color, y_err=None, style='MC'):
    bin_center = (binning[1:] + binning[:-1]) / 2.
    bin_width = np.diff(binning)/2.
    if style == 'MC':
        __plot_mc_style(ax, hist, bin_center, binning, color, label)


def __plot_data_style(hist, bin_center):
    pass

def __plot_mc_style(ax, hist, bin_center, binning, color, label):
    try:
        ax.hist(bin_center,
            bins=binning,
            weights=hist,
            histtype='step',
            color=color,
            lw=lw,
            label=label)
    except:
        print(binning)
        exit()

def __plot_uncertainties__(hist, bin_center, bin_width, y_err):
    pass


def save_fig(fig, name, tight=True):
    if tight:
        fig.savefig(name+'.png', bbox_inches='tight')
    else:
        fig.savefig(name+'.png')
    plt.close(fig)
