#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import scipy.stats.distributions as sc_dist

import os

import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from tqdm import tqdm

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

get_color.pointer = -1

uncertainties_cycle = ['viridis_r',
                       'plasma_r',
                       'viridis_r',
                       'magma_r',
                       'inferno_r']


def get_cmap():
    get_cmap.pointer += 1
    if get_cmap.pointer >= len(uncertainties_cycle):
        get_cmap.pointer = 0
    return uncertainties_cycle[get_cmap.pointer]

get_cmap.pointer = -1

LW = 2.
MS = '5'
ZORDER = 2
BORDER_OFFSET = 0.1
RATIO_LIMIT = 1e-3


def plot(output,
         title,
         components,
         binnings,
         plotting_keys,
         transformed_keys,
         alphas,
         plot_ratios=False):
    legend_objects = []
    legend_labels = []
    n_obs = len(plotting_keys)
    for i, o in enumerate(plotting_keys):
        if plot_ratios:
            fig = plt.figure(0)
            ratio_components = [c for c in components
                                if c.uncertainties is not None]
            height_main = 0.5 + 0.5 * (1 / (1 + len(ratio_components)))

            gs = GridSpec(1, 1)
            ax = fig.add_subplot(gs[0])
            plt.setp(ax.get_xticklabels(), visible=False)
            gs.update(left=0.+BORDER_OFFSET,
                      right=1-BORDER_OFFSET,
                      top=1-BORDER_OFFSET,
                      bottom=1-height_main)
            gs_ratio = GridSpec(len(ratio_components)*2, 1)
            ax_ratio = {}
            for k, c in enumerate(ratio_components):
                ax_ratio[c.name] = [fig.add_subplot(gs_ratio[k*2]),
                                    fig.add_subplot(gs_ratio[k*2+1])]

            gs_ratio.update(left=0.+BORDER_OFFSET,
                            right=1-BORDER_OFFSET,
                            top=1-height_main,
                            bottom=0.+BORDER_OFFSET,
                            hspace=0.,
                            wspace=0.)
        else:
            fig, ax = plt.subplots()
        binning = binnings[i]
        ax.set_xlim(binning[0], binning[-1])
        ax.set_yscale("log", nonposy='clip')
        for j, c in enumerate(components):
            hist = c.hists[i, :]
            if not all(hist == 0.):
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
                        if i == 0:
                            legend_objects.extend(obj)
                            legend_labels.extend(lab)
        if plot_ratios:
            for key in ax_ratio.keys():
                ref_c = components[components.index(key)]
                ref_hist = ref_c.hists[i, :]
                ax_ratio_c = ax_ratio[key]
                ax_plus = ax_ratio_c[0]
                ax_minus = ax_ratio_c[1]
                ax_plus.set_ylim(ymin=1., ymax=RATIO_LIMIT)
                ax_plus.xaxis.set_ticklabels([])
                ax_minus.set_ylim(ymin=RATIO_LIMIT, ymax=1.)
                format_axis(ax_plus, binning)
                format_axis(ax_minus, binning)
                ref_uncerts = ref_c.uncert_ratio
                plot_uncertainties_ratio(fig,
                                         ax_ratio_c,
                                         ref_c.uncert_ratio[i],
                                         binning,
                                         ref_c.label,
                                         ref_c.color,
                                         ref_c.cmap,
                                         alphas)
                for c in components:
                    hist = c.hists[i, :]
                    if c.name not in ax_ratio.keys() and c.ctype == 'Data':
                        plot_data_ratio(fig,
                                        ax_ratio_c,
                                        c.uncert_ratio[ref_c.name][i],
                                        binning,
                                        c.label,
                                        c.color)
            x_label_ax = ax_minus
            ax.locator_params(axis='y', tight=True)
            #plt.setp(ax_r.get_xticklabels(), visible=True)
        else:
            x_label_ax = ax
        ax.legend(legend_objects, legend_labels,
                  handler_map=le.handler_mapper,
                  loc='best',
                  prop={'size': 6})
        x_label_ax.set_xlabel(transformed_keys[i])
        ax.set_ylabel('# Entries [Hz]')
        fig.suptitle(title, fontsize=14)
        save_fig(fig, os.path.join(output, plotting_keys[i]), tight=False)


def format_axis(ax, binning):
    ax.set_yscale("log")
    ax.set_xlim(binning[0], binning[-1])
    ax.yaxis.set_ticks(np.logspace(-1,RATIO_LIMIT,3))
    major_ticks = np.logspace(-1,RATIO_LIMIT,3)
    minor_ticks = np.logspace(-1,RATIO_LIMIT,5)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.yaxis.grid(which='both')
    ax.yaxis.grid(which='minor', alpha=0.2)
    ax.yaxis.grid(which='major', alpha=0.5)










def plot_uncertainties_ratio(fig,
                             ax_ratio,
                             uncerts_ratio,
                             binning,
                             label,
                             color,
                             cmap,
                             alphas):
    [ax_plus, ax_minus] = ax_ratio
    n_alphas = len(alphas)
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.1, 0.9, len(alphas)))
    legend_entries = []
    legend_labels = []
    legend_entries.append(le.UncertObject(colors, color))
    legend_labels.append(label)
    for i, (c, a) in enumerate(zip(colors[::-1], alphas[::-1])):
        j = n_alphas - i - 1
        lower_limit = np.append(uncerts_ratio[0, j, 0], uncerts_ratio[:, j, 0])
        upper_limit = np.append(uncerts_ratio[0, j, 1], uncerts_ratio[:, j, 1])
        ax_plus.fill_between(
                             binning,
                             np.ones_like(upper_limit),
                             upper_limit,
                             step='pre',
                             color=c,
                             zorder=ZORDER)
        ax_minus.fill_between(
                             binning,
                             np.ones_like(lower_limit),
                             lower_limit,
                             step='pre',
                             color=c,
                             zorder=ZORDER)



def plot_data_ratio(fig,
                    ax_ratio,
                    uncerts_ratio,
                    binning,
                    label,
                    color):
    [ax_plus, ax_minus] = ax_ratio
    bin_center = (binning[1:] + binning[:-1]) / 2.
    null_mask = np.isnan(uncerts_ratio)
    neg_inf = np.logical_and(np.isinf(uncerts_ratio), uncerts_ratio < 0)
    pos_inf = np.logical_and(np.isinf(uncerts_ratio), uncerts_ratio > 0)

    bin_center = bin_center[~null_mask]
    uncerts_ratio = uncerts_ratio[~null_mask]
    plus_mask = uncerts_ratio > 0
    minus_mask = uncerts_ratio < 0

    markeredgecolor = 'k'
    markerfacecolor = color
    plot_data_ratio_part(fig, ax_plus,
                         bin_center[plus_mask],
                         uncerts_ratio[plus_mask],
                         color,
                         bot=False)
    plot_data_ratio_part(fig, ax_minus,
                         bin_center[minus_mask],
                         np.absolute(uncerts_ratio[minus_mask]),
                         color,
                         bot=True)
    plot_inf_marker(fig, ax_minus,
                    binning,
                    ~neg_inf,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    bot=True)

    plot_inf_marker(fig, ax_plus,
                    binning,
                    ~pos_inf,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    bot=False)

def plot_data_ratio_part(fig, ax, x, y, color, bot=True):
    markeredgecolor = 'k'
    markerfacecolor = color
    infinite_mask = np.isinf(y)
    ax.plot(x[~infinite_mask],
            y[~infinite_mask],
            ls='', ms=MS,
            mew=1.,
            marker='o',
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            zorder=ZORDER+100,
            clip_on=False)


def generate_ticks(y_0, y_min, max_ticks_per_side=5):
    y_min = np.floor(y_min)
    y_0_log = np.log10(y_0)

    tick_pos = []

    n_ticks = 1
    tick_pos.append(y_0_log)
    if y_0_log != np.floor(y_0_log):
        tick_pos.append(np.floor(y_0_log))
        n_ticks += 2
    while tick_pos[-1] > y_min:
        tick_pos.append(tick_pos[-1]-1)
        n_ticks += 2
    n_ticks_per_side = (n_ticks-1)/2
    mayor_step_size = np.ceil(n_ticks_per_side/max_ticks_per_side)
    tick_pos_mapped, _, _ = map_ratio(np.power(10, tick_pos),
                                                y_min=y_min,
                                                y_0=y_0)
    mayor_ticks = []
    mayor_ticks_labels = []

    minor_ticks = []
    minor_ticks_labels = []
    mayor_tick_counter = 0
    for i, [p, l] in enumerate(zip(tick_pos_mapped, tick_pos)):
        lab = 10**l
        lab = u'10$^{\mathregular{%d}}$' % l
        if i == 0:
            mayor_ticks_labels.append(lab)
            mayor_ticks.append(0)
        else:
            if mayor_tick_counter == mayor_step_size:
                mayor_ticks.extend([p*-1, p])
                mayor_ticks_labels.extend([lab, lab])
                mayor_tick_counter = 0
            else:
                minor_ticks.extend([p*-1, p])
                minor_ticks_labels.extend([lab, lab])
                mayor_tick_counter += 1

    return mayor_ticks_labels, mayor_ticks, minor_ticks_labels, minor_ticks

def map_ratio(y_values, y_min=None, y_0=1.):
    flattened_y = y_values.reshape(np.prod(y_values.shape))
    infinite = np.isinf(flattened_y)
    finite = np.isfinite(flattened_y)
    finite_y = flattened_y[finite]
    finite_y[finite_y > y_0] = np.NaN
    plus_mask = finite_y > 0
    minus_mask = finite_y < 0
    finite_y = np.absolute(finite_y)
    finite_y[plus_mask] = np.log10(finite_y[plus_mask])
    finite_y[minus_mask] = np.log10(finite_y[minus_mask])
    if y_min is None:
        y_min = min(np.min(finite_y[plus_mask]),
                       np.min(finite_y[minus_mask]))
    finite_y /= np.absolute(y_min)
    finite_y[finite_y > 1] = np.inf
    finite_y[finite_y < -1] = -np.inf
    finite_y[minus_mask] *= -1
    tranformed_values = np.zeros_like(flattened_y)
    tranformed_values[:] = np.NaN
    tranformed_values[finite] = finite_y
    tranformed_values[infinite] = flattened_y[infinite]
    return tranformed_values.reshape(y_values.shape)*-1, y_0, y_min


def plot_data_ratio_mapped(fig,
                           ax,
                           uncerts_ratio,
                           binning,
                           label,
                           color):
    bin_center = (binning[1:] + binning[:-1]) / 2.
    finite_mask = np.isfinite(uncerts_ratio)
    neg_inf = np.isneginf(uncerts_ratio)
    pos_inf = np.isposinf(uncerts_ratio)

    bin_center = bin_center[finite_mask]
    uncerts_ratio = uncerts_ratio[finite_mask]

    markeredgecolor = 'k'
    markerfacecolor = color
    plot_data_ratio_part(fig, ax,
                         bin_center,
                         uncerts_ratio,
                         color,
                         bot=False)
    plot_inf_marker(fig, ax,
                    binning,
                    ~neg_inf,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    bot=True)
    plot_inf_marker(fig, ax,
                    binning,
                    ~pos_inf,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    bot=False)


def plot_uncertainties_ratio_mapped(fig,
                             ax_ratio,
                             uncerts_ratio,
                             binning,
                             y_0,
                             y_min,
                             label,
                             color,
                             cmap,
                             alphas):
    n_alphas = len(alphas)
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.1, 0.9, len(alphas)))
    for i, (c, a) in enumerate(zip(colors[::-1], alphas[::-1])):
        j = n_alphas - i - 1
        lower_limit = uncerts_ratio[:, j, 0]
        upper_limit = uncerts_ratio[:, j, 1]
        upper_limit[np.isinf(upper_limit)] = 0
        lower_limit[np.isinf(lower_limit)] = -1
        ax_ratio.fill_between(
                             binning,
                             upper_limit,
                             lower_limit,
                             step='pre',
                             color=c)



def plot_data_style(fig, ax, hist, binning, label, color):
    zero_mask = hist > 0
    bin_center = (binning[1:] + binning[:-1]) / 2.

    markeredgecolor = 'k'
    markerfacecolor = color
    ax.plot(bin_center[zero_mask],
            hist[zero_mask],
            ls='', ms=MS,
            mew=1.,
            marker='o',
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            zorder=ZORDER+1)
    plot_inf_marker(fig, ax,
                    binning,
                    zero_mask,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
    return le.DataObject(markerfacecolor,
                         markeredgecolor,
                         markerfacecolor,
                         markeredgecolor), label


def plot_inf_marker(fig, ax, binning, zero_mask, markeredgecolor='k',
                    markerfacecolor='none', bot=True):
    patches = []
    radius = 0.008
    bbox = ax.get_position()
    x_0 = bbox.x0
    width = bbox.x1 - bbox.x0
    if bot:
        y0 = bbox.y0 + radius
        orientation = np.pi
    else:
        y0 = bbox.y1 - radius
        orientation = 0
    bin_center = (binning[1:] + binning[:-1]) / 2
    binning_width = binning[-1] - binning[0]
    bin_0 = binning[0]
    for bin_i, mask_i in zip(bin_center, zero_mask):
        if not mask_i:
            x_i = ((bin_i - bin_0) / binning_width * width) + x_0
            patches.append(mpatches.RegularPolygon([x_i, y0], 3,
                                                   radius=radius,
                                                   orientation=orientation,
                                                   facecolor=markerfacecolor,
                                                   edgecolor=markeredgecolor,
                                                   transform=fig.transFigure,
                                                   figure=fig,
                                                   linewidth=1.,
                                                   zorder=ZORDER+1))
    fig.patches.extend(patches)


def plot_mc_style(fig, ax, hist, binning, label, color, linewidth=None):
        if linewidth is None:
            linewidth = LW
        try:
          obj, = ax.plot(binning,
                        np.append(hist[0], hist),
                        drawstyle='steps-pre',
                        lw=linewidth,
                        c=color,
                        label=label,
                        zorder=ZORDER)
        except ValueError:
            print(hist)
            exit()
        return obj, label


def plot_uncertainties(fig, ax, hist, uncert, binning,
                       label, color, cmap, alphas):
    _, _ = plot_mc_style(fig,
                         ax,
                         hist,
                         binning,
                         label,
                         color,
                         linewidth=LW - 1.)
    ax.set_xlim(binning[0], binning[-1])
    n_alphas = len(alphas)
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.1, 0.9, len(alphas)))
    legend_entries = []
    legend_labels = []
    legend_entries.append(le.UncertObject(colors, color))
    legend_labels.append(label)
    for i, (c, a) in enumerate(zip(colors[::-1], alphas[::-1])):
        j = n_alphas - i - 1
        lower_limit = uncert[:, j, 0] * hist
        upper_limit = uncert[:, j, 1] * hist
        ax.fill_between(
            binning,
            np.append(lower_limit[0], lower_limit),
            np.append(upper_limit[0], upper_limit),
            step='pre',
            color=c,
            zorder=ZORDER)
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
