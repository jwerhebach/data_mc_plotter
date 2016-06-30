#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os
import warnings

from tqdm import tqdm

import numpy as np
import tables

from matplotlib import pyplot as plt

import data_handler as dh
from .aggregation_math import aggregate
from .aggarwal_err import calc_limits
import plot_funcs
import scripts.config_parser_helper as ch

class ComparisonPlotter:
    def __init__(self,
                 components,
                 id_keys,
                 match=False,
                 n_bins=50):
        self.components = components
        self.id_dict = ch.split_obs_str(id_keys)
        assert len(self.id_dict.keys()) == 1, \
            'id_keys have to be from one table!'
        self.id_cols = self.id_dict[self.id_dict.keys()[0]][0]
        self.match = match
        self.n_bins = n_bins
        self.livetime = 0.

        for i, c in enumerate(components):
            c.init_component(self.id_dict)
            if c.ctype == 'Data':
                self.livetime += c.livetime
        self.data_components = [c for c in self.components
                                if c.aggregation is None]
        self.plotting_components = [c for c in self.components
                                    if c.show]
        self.cmds = []
        for c in self.plotting_components:
            if c.aggregation is None:
                self.cmds.append(c.name)
            else:
                self.cmds.append(c.aggregation.cmd)

    def plot(self, observables, outpath, uncertainties=''):
        observables = ch.split_obs_str(observables)
        print('Fetching Observable infos')
        n_obs = np.sum([len(observables[k][0]) for k in observables.keys()])
        list_data = [str(c) for c in self.data_components]
        list_plot = [str(c) for c in self.plotting_components]
        with tqdm(total=n_obs, unit='Observables') as pbar:
            for table_key, [cols, trans] in observables.iteritems():
                hists = np.empty((len(cols),
                                  len(self.data_components),
                                  self.n_bins))
                binnings = np.empty((len(cols), self.n_bins+1))
                trans_obs = []
                cols_mask = []
                for i, comp in enumerate(self.data_components):
                    all_values = comp.get_values(table_key, cols)
                    for j, [c, t] in enumerate(zip(cols, trans)):
                        obs_key = '%s.%s' % (table_key, c)
                        if i == 0:
                                cols_mask.append(True)
                        if cols_mask[j]:
                            vals, weights = dh.filter_nans(all_values[:, j],
                                                           comp.weight)
                            vals = dh.transform_values(vals, t)
                            transformed_obs_key = dh.transform_obs(obs_key, t)
                            trans_obs.append(transformed_obs_key)
                            if i == 0:
                                if len(vals) == 0:
                                    cols_mask[j] = False
                                else:
                                    binning = self.min_max_binning(vals)
                                    if binning is None:
                                        cols_mask[j] = False
                                    else:
                                        binnings[j] = binning
                            if cols_mask[j]:
                                hist = np.histogram(vals,
                                                    bins=binnings[j],
                                                    weights=weights)[0]
                                hists[j, i, :] = hist * self.livetime
                hists = hists[np.where(cols_mask)[0], :, :]
                plotting_hists = aggregate(hists,
                                           self.cmds,
                                           list_data,
                                           list_plot)
                plotting_cols = [c for i, c in enumerate(cols) if cols_mask[i]]
                binnings = binnings[np.where(cols_mask)[0]]
                for i, col in enumerate(plotting_cols):
                    obs_key = '%s.%s' % (table_key, col)
                    f, ax = plt.subplots()
                    ax.set_yscale('log',)
                    for j, comp in enumerate(self.plotting_components):
                        plot_funcs.plot_hist(ax,
                                             label=comp.label,
                                             hist=plotting_hists[i, j, :],
                                             binning=binnings[i],
                                             color=comp.color,
                                             style='MC')
                    plot_funcs.save_fig(f, obs_key)
                pbar.update(len(cols))

    def get_possiblites(self,
                        outpath,
                        check_all=False,
                        blacklist=None):
        if isinstance(blacklist, dict):
            blacklist_cols = blacklist['cols']
            blacklist_cols.extend(self.id_cols)
            blacklist_obs = blacklist['obs']
            blacklist_tabs = blacklist['tabs']
        else:
            blacklist_cols = self.id_cols
            blacklist_tabs = []
            blacklist_obs = []
        blacklist = [blacklist_tabs, blacklist_cols, blacklist_obs]
        obs_set = set()
        for i, c in enumerate(self.data_components):
            component_set = c.get_observables(blacklist, check_all)
            if i == 0:
                obs_set = component_set
            else:
                obs_set = obs_set.intersection(component_set)
        sorted_list = sorted(obs_set)
        with open(os.path.join(outpath, 'observable.txt'), 'wb') as f:
            f.write('[')
            for i, observable in enumerate(sorted_list):
                if i == 0:
                    f.write('%s' % observable)
                else:
                    f.write(',\n\t%s' % observable)
            f.write(']')
        return sorted_list

    def min_max_binning(self, vals):
        max_val = np.max(vals)
        min_val = np.min(vals)
        if max_val == min_val:
            return None
        else:
            return np.linspace(min_val, max_val, self.n_bins + 1)




