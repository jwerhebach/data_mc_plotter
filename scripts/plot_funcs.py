#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import os

import legend_entries as le


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
                if c.uncertainties is None:
                    obj, lab = plot_mc_style(fig,
                                             ax,
                                             hist,
                                             binning,
                                             c.label,
                                             c.color)
                    if i == 0:
                        legend_objects.append(obj)
                        legend_labels.append(lab)
                else:
                    uncert = c.uncertainties[i, :]
                    obj, lab = plot_uncertainties(fig,
                                                  ax,
                                                  hist,
                                                  uncert,
                                                  binning,
                                                  c.label,
                                                  c.color,
                                                  c.cmap,
                                                  alphas)
                    _, _ = plot_mc_style(fig,
                                         ax,
                                         hist,
                                         binning,
                                         c.label,
                                         c.color)
                    if i == 0:
                        legend_objects.extend(obj)
                        legend_labels.extend(lab)
        ax.legend(legend_objects, legend_labels, handler_map=le.handler_mapper)
        save_fig(fig, os.path.join(output, obs_keys[i]), tight=True)


def plot_data_style(fig, ax, hist, binning, label, color):
    zero_mask = hist > 0
    bin_center = (binning[1:] + binning[:-1]) / 2.

    markeredgecolor = 'k'
    markerfacecolor = color

    ax.plot(bin_center[zero_mask],
            hist[zero_mask],
            ls='', ms=ms,
            mew=lw - 0.5,
            marker='o',
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor)
    plot_zero_marker(fig, ax,
                     bin_center[~zero_mask],
                     [binning[0], binning[-1]],
                     markerfacecolor=markerfacecolor,
                     markeredgecolor=markeredgecolor)
    return le.DataObject(markerfacecolor,
                         markeredgecolor,
                         markerfacecolor,
                         markeredgecolor), label


def plot_zero_marker(fig, ax, x, x_lims, markeredgecolor='k',
                     markerfacecolor='none'):
    patches = []
    radius = 0.008
    bbox = ax.get_position()
    x_0 = bbox.x0
    width = bbox.x1 - bbox.x0
    y0 = bbox.y0
    for x_i in x:
        x_i = (x_i / np.diff(x_lims) * width) + x_0
        patches.append(mpatches.RegularPolygon([x_i, y0 + radius], 3,
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


def plot_uncertainties(fig, ax, hist, uncert, binning,
                       label, color, cmap, alphas):
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.1, 0.9, len(alphas)))
    legend_entries = []
    legend_labels = []
    legend_entries.append(le.UncertObject(colors, color))
    legend_labels.append(label)
    for i, (c, a) in enumerate(zip(colors[::-1], alphas[::-1])):
        lower_limit = uncert[:, i, 0]
        upper_limit = uncert[:, i, 1]
        ax.fill_between(
            binning,
            np.append(lower_limit[0], lower_limit),
            np.append(upper_limit[0], upper_limit),
            step='pre',
            color=c)
    for i, (c, a) in enumerate(zip(colors, alphas)):
        legend_entries.append(le.UncertObject_single(c))
        legend_labels.append('      %.1f%% Uncert.' % (a * 100.))
    return legend_entries, legend_labels


def save_fig(fig, name, tight=True):
    if tight:
        fig.savefig(name + '.png', bbox_inches='tight')
    else:
        fig.savefig(name + '.png')
    plt.close(fig)
