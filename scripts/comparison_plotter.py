#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os

from tqdm import tqdm

import numpy as np

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

    def plot(self, title, observables, outpath, alphas=[]):
        observables = ch.split_obs_str(observables)
        print('Fetching Observable infos')
        n_obs = np.sum([len(observables[k][0]) for k in observables.keys()])
        list_data = [str(c) for c in self.data_components]
        list_plot = [str(c) for c in self.plotting_components]
        hists = np.empty((len(self.data_components), n_obs, self.n_bins))
        binnings = np.empty((n_obs, self.n_bins + 1))
        trans_obs = []
        obs_keys = []
        cols_mask = []
        finished_cols = 0
        with tqdm(total=n_obs, unit='Observables') as pbar:
            for table_key, [cols, trans] in observables.iteritems():
                for i, comp in enumerate(self.data_components):
                    all_values = comp.get_values(table_key, cols)
                    for j, [c, t] in enumerate(zip(cols, trans)):
                        current_col = finished_cols + j
                        obs_key = '%s.%s' % (table_key, c)
                        if i == 0:
                                obs_keys.append(obs_key)
                                cols_mask.append(True)
                                transformed_obs_key = dh.transform_obs(
                                    obs_key, t)
                                trans_obs.append(transformed_obs_key)
                        if cols_mask[current_col]:
                            vals, weights = dh.filter_nans(all_values[:, j],
                                                           comp.weight)
                            vals, weights = dh.transform_values(t, vals,
                                                                weights)
                            if i == 0:
                                if len(vals) == 0:
                                    cols_mask[current_col] = False
                                else:
                                    binning = self.min_max_binning(vals)
                                    if binning is None:
                                        cols_mask[current_col] = False
                                    else:
                                        binnings[current_col] = binning
                            if cols_mask[current_col]:
                                print(binnings[j])
                                hist = np.histogram(vals,
                                                    bins=binnings[j],
                                                    weights=weights)[0]
                                if comp.ctype == 'MC':
                                    hists[i, current_col, :] = hist * self.livetime
                                else:
                                    hists[i, current_col, :] = hist
                                if any(np.isnan(hists[i, current_col, :])):
                                    print('ALARM')
                                    print(obs_key)
                finished_cols += len(cols)
                pbar.update(len(cols))
        hists = hists[:, np.where(cols_mask)[0], :]
        plotting_hists = aggregate(hists,
                                   self.cmds,
                                   list_data,
                                   list_plot)
        plotting_keys = [c for i, c in enumerate(obs_keys)
                         if cols_mask[i]]
        transformed_keys = [c for i, c in enumerate(trans_obs)
                            if cols_mask[i]]
        binnings = binnings[np.where(cols_mask)[0]]
        for i, comp in enumerate(self.plotting_components):
            comp.hists = plotting_hists[i, :, :]
            if comp.calc_uncertainties:
                if len(alphas) > 0:
                    alphas = sorted(alphas)
                    comp.uncertainties = calc_limits(comp.hists,
                                                     alphas)
        plotting_hists /= self.livetime
        plot_funcs.plot(outpath,
                        title,
                        self.plotting_components,
                        binnings,
                        plotting_keys,
                        plotting_keys,
                        transformed_keys,
                        alphas)

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
