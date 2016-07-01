#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import os

class DataObject(object):
    def __init__(self, fc_t='w', ec_t='k', fc_c='w', ec_c='k'):
        self.fc_t = fc_t
        self.ec_t = ec_t
        self.fc_c = fc_c
        self.ec_c = ec_c

class data_handler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        scale = fontsize / 22
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        radius = height/2 * scale
        xt_0 = x0 + width - height/2
        xt_l = xt_0 - radius
        xt_r = xt_0 + radius
        yt_0 = y0 + height/2
        yt_t = yt_0 + radius
        yt_b = yt_0 - radius

        xc_0 = x0 + height/2 + radius
        yc_0 = y0 + height/2 * (1 - scale) + radius

        triangle = np.asarray([[xt_l, yt_t],
                               [xt_0, yt_b],
                               [xt_r, yt_t]])

        patch_tri = mpatches.Polygon(triangle,
                                     facecolor=orig_handle.fc_t,
                                     edgecolor=orig_handle.ec_t,
                                     transform=handlebox.get_transform())
        patch_tri = mpatches.RegularPolygon(
            [xt_0, yc_0],
            3,
            radius=radius*1.5,
            orientation=np.pi,
            facecolor=orig_handle.fc_t,
            edgecolor=orig_handle.ec_t,
            transform=handlebox.get_transform())
        patch_circ = mpatches.Circle([xc_0, yc_0], radius,
                                     facecolor=orig_handle.fc_c,
                                     edgecolor=orig_handle.ec_c,
                                     transform=handlebox.get_transform())
        handlebox.add_artist(patch_tri)
        handlebox.add_artist(patch_circ)
        return patch_circ


handler_mapper = {DataObject: data_handler()}

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

color_cycle = [(r / 255., g / 255., b / 255) for r, g, b in color_cycle]


def get_color():
    get_color.pointer += 1
    if get_color.pointer >= len(color_cycle):
        get_color.pointer = 0
    return color_cycle[get_color.pointer]

get_color.pointer = 0

uncertainties_cycle = ['viridis',
                       'plasma',
                       'magma',
                       'inferno']

def get_cmap():
    get_cmap.pointer += 1
    if get_cmap.pointer >= len(uncertainties_cycle):
        get_cmap.pointer = 0
    return uncertainties_cycle[get_cmap.pointer]

get_cmap.pointer = 0

lw = 2.
ms = '5'
edgecolor_data = 'k'


def plot(output,
         components,
         binnings,
         plotting_keys,
         obs_keys,
         transformed_keys,
         alphas):
    handler_map_dict = {}
    legend_objects = []
    legend_labels = []
    for i, o in enumerate(obs_keys):

        fig, ax = plt.subplots()
        binning = binnings[i]
        ax.set_xlim(binning[0], binning[-1])
        ax.set_yscale("log", nonposy='clip')
        for j, c in enumerate(components):
            hist = c.hists[i, :]
            if c.ctype == 'Data':
                obj, lab = plot_data_style(fig,
                                           ax,
                                           hist,
                                           binning,
                                           c.label,
                                           c.color)
                if i == 0:
                    legend_objects.append(obj)
                    legend_labels.append(lab)
            if c.ctype == 'MC':
                obj, lab = plot_mc_style(fig,
                                           ax,
                                           hist,
                                           binning,
                                           c.label,
                                           c.color)
                if i == 0:
                    legend_objects.append(obj)
                    legend_labels.append(lab)
        ax.legend(legend_objects, legend_labels, handler_map=handler_mapper)
        save_fig(fig, os.path.join(output, obs_keys[i]), tight=True)

def plot_data_style(fig, ax, hist, binning, label, color):
    zero_mask = hist > 0
    bin_center = (binning[1:] + binning[:-1]) / 2.

    markeredgecolor = 'k'
    markerfacecolor = color

    ax.plot(bin_center[zero_mask],
            hist[zero_mask],
            ls='', ms=ms,
            mew=lw-0.5,
            marker='o',
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor)
    plot_zero_marker(fig, ax,
                     bin_center[~zero_mask],
                     [binning[0], binning[-1]],
                     markerfacecolor=markerfacecolor,
                     markeredgecolor=markeredgecolor)
    return DataObject(markerfacecolor,
                      markeredgecolor,
                      markerfacecolor,
                      markeredgecolor), label


def plot_zero_marker(fig, ax, x, x_lims, markeredgecolor='k',
                     markerfacecolor='none'):
    patches = []
    radius = 0.008
    bbox = ax.get_position()
    x_0 = bbox.x0
    width = bbox.x1-bbox.x0
    y0 = bbox.y0
    for x_i in x:
        x_i = (x_i/np.diff(x_lims) * width)+ x_0
        patches.append(mpatches.RegularPolygon([x_i, y0+radius], 3,
                                               radius=radius,
                                               orientation=np.pi,
                                               facecolor=markerfacecolor,
                                               edgecolor=markeredgecolor,
                                               transform=fig.transFigure,
                                               figure=fig))
    fig.patches.extend(patches)


def plot_mc_style(fig, ax, hist, binning, label, color):
        obj, = ax.plot(binning,
                       np.append(hist[0], hist),
                       drawstyle='steps-pre',
                       lw=lw,
                       c=color,
                       label=label)
        return obj, label

def plot_uncertainties(hist, bin_center, bin_width, y_err):
    colors = cmap(np.linspace(0.1, 0.9, len(alphas)))


def save_fig(fig, name, tight=True):
    if tight:
        fig.savefig(name+'.png', bbox_inches='tight')
    else:
        fig.savefig(name+'.png')
    plt.close(fig)
