#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

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


def plot(output,
         title,
         components,
         binnings,
         plotting_keys,
         obs_keys,
         transformed_keys,
         alphas,
         plot_ratios=False):
    print(plot_ratios)
    legend_objects = []
    legend_labels = []
    n_obs = len(obs_keys)
    print('Plot Observables')
    with tqdm(total=n_obs, unit='Observables') as pbar:
        for i, o in enumerate(obs_keys):
            if plot_ratios:
                fig = plt.figure(0)
                ratio_components = [c for c in components
                                    if c.uncertainties is not None]
                height = 1 / 2 / (1 + len(ratio_components))
                height_ratios = [0.5 + height]
                ax_ratio = {}
                for c in ratio_components:
                    height_ratios.append(height)
                gs = GridSpec(1 + len(ratio_components), 1,
                              height_ratios=height_ratios)
                gs.update(hspace=0.07)
                ax = plt.subplot(gs[0])
                for k, c in enumerate(ratio_components):
                    ax_ratio[c.name] = {}
                    ax_ratio[c.name] = plt.subplot(gs[1 + k], sharex=ax)
                plt.setp(ax.get_xticklabels(), visible=False)
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
                    ax_r = ax_ratio[key]
                    ref_uncerts = ref_c.uncertainties[i, :]
                    _, _ = plot_uncertainties(fig,
                                              ax_r,
                                              np.ones_like(ref_hist),
                                              ref_uncerts,
                                              binning,
                                              ref_c.label,
                                              ref_c.color,
                                              ref_c.cmap,
                                              alphas)
                    ax_r.set_ylabel('Data/MC')
                    tick_locator = MaxNLocator(nbins=4)
                    ax_r.get_yaxis().set_major_locator(tick_locator)
                    plt.setp(ax_r.get_xticklabels(), visible=False)
                    for c in components:
                        hist = c.hists[i, :]
                        if c.name not in ax_ratio.keys() and c.ctype == 'Data':
                            plot_data_ratio(fig, ax_r, hist, ref_hist,
                                            binning, c.label, c.color)
                x_label_ax = ax_r
                ax.locator_params(axis='y', tight=True)
                plt.setp(ax_r.get_xticklabels(), visible=True)
            else:
                x_label_ax = ax
            ax.legend(legend_objects, legend_labels,
                      handler_map=le.handler_mapper,
                      loc='best',
                      prop={'size': 11})
            x_label_ax.set_xlabel(transformed_keys[i])
            ax.set_ylabel('# Entries [Hz]')
            fig.suptitle(title, fontsize=14)
            save_fig(fig, os.path.join(output, obs_keys[i]), tight=False)
            pbar.update(1)


def plot_data_ratio(fig, ax, hist, ref_hist, binning, label, color):
    not_zero_mask = hist > 0
    not_zero_mask_ref = ref_hist > 0
    finite_ratio_mask = np.logical_and(not_zero_mask, not_zero_mask_ref)
    zero_ratio_mask = np.logical_and(~not_zero_mask, not_zero_mask_ref)
    infinite_ratio_mask = np.logical_and(not_zero_mask, ~not_zero_mask_ref)
    bin_center = (binning[1:] + binning[:-1]) / 2.

    markeredgecolor = 'k'
    markerfacecolor = color
    ax.plot(bin_center[finite_ratio_mask],
            hist[finite_ratio_mask] / ref_hist[finite_ratio_mask],
            ls='', ms=MS,
            mew=1.,
            marker='o',
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            zorder=ZORDER+1)
    plot_inf_marker(fig, ax,
                    binning,
                    ~zero_ratio_mask,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    bot=True)
    plot_inf_marker(fig, ax,
                    binning,
                    ~infinite_ratio_mask,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    bot=False)


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
