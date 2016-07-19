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

percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]


class ComparisonPlotter:
    def __init__(self,
                 components,
                 id_keys,
                 match=False,
                 autotransform=False,
                 n_bins=50,
                 alphas=[0.682689492, 0.9, 0.99],
                 plot_ratios=True):
        self.components = components
        id_dict = ch.split_obs_str(id_keys)
        assert len(id_dict.keys()) == 1, \
            'id_keys have to be from one table!'
        self.id_table = id_dict.keys()[0]
        self.id_cols = id_dict[self.id_table][0]
        self.match = match
        self.autotransform = autotransform
        self.n_bins = n_bins
        self.alphas = sorted(alphas)

        self.data_livetime = 0.
        self.n_events_data = 0

        for i, c in enumerate(components):
            c.init_component(self.id_table, self.id_cols)
            if c.ctype == 'Data':
                self.data_livetime += c.livetime
                self.n_events_data = c.get_nevents()
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
        self.plot_ratios = plot_ratios

    def auto_scale(self, scaling_list):
        for c_i in scaling_list:
            index = self.components.index(c_i)
            curr_component = self.components[index]
            sum_w_i = curr_component.get_nevents(weighted=True)
            n_events_i = sum_w_i * self.data_livetime
            scaling_factor = self.n_events_data / n_events_i
            curr_component.set_scaling(scaling_factor)
            print('Scaling for \'%s\' set to: %.4f' % (c_i, scaling_factor))

    def fetch_data_and_plot(self, title, observables, outpath):
        observables = ch.split_obs_str(observables)
        print('Fetching Observable infos')
        n_obs = np.sum([len(observables[k][0]) for k in observables.keys()])
        list_data = [str(c) for c in self.data_components]
        list_plot = [str(c) for c in self.plotting_components]
        n_events = np.asarray(
            [c.get_nevents(weighted=True) for c in self.data_components])

        with tqdm(total=n_obs, unit='Observable') as pbar:
            for table_key, [cols, trafos] in observables.iteritems():
                for i, comp in enumerate(self.data_components):
                    comp_values = comp.get_values(table_key, cols)
                    comp_values.replace([np.inf, -np.inf], np.nan,
                                         inplace=True)
                    comp_values[comp_values.abs() > 1e20] = np.nan
                    if i == 0:
                        all_values = comp_values
                    else:
                        all_values = all_values.append(comp_values)
                binning_dict = {}
                if self.autotransform:
                    drops = []
                    for i, c in enumerate(cols):
                        series = all_values[c]
                        min_val = serius.min()
                        max_val = series.max()
                        if not np.isfinite(mins[c]):
                            drops.append(c)
                        elif np.isclose(mins[c], maxs[c]):
                            drops.append(c)
                        if issubclass(series.dtype.type, np.integer):
                            diff = max_val - min_val
                            if diff < self.n_bins*2:
                                binning_dict[c] = np.linspace(min_val-0.5,
                                                              max_val+0.5,
                                                              diff+1)
                            elif diff > 1e3:
                                binning_dict[c] = np.linspace(min_val-0.5,
                                                              max_val+0.5,
                                                              diff+1)
                    if len(drops) > 0:
                        all_values.drop(drops, axis=1, inplace=True)
                else:
                    plot_labels = {}
                    for c, t in zip(cols, trafos):
                        obs_key = '%s.%s' % (table_key, c)
                        all_values[c], c_t = transform_values(
                            t, all_values[c], obs_key)
                        plot_labels[c] = [obs_key, c_t]
                    mins = all_values.min()
                    maxs = all_values.max()
                    drops = []
                    for c in cols:
                        if not np.isfinite(mins[c]):
                            drops.append(c)
                        elif np.isclose(mins[c], maxs[c]):
                            drops.append(c)
                        else:
                            offset = (maxs[c] - mins[c]) * 1e-3
                            binning_dict[c] = np.linspace(mins[c],
                                                          maxs[c]+offset,
                                                          self.n_bins+1)
                    if len(drops) > 0:
                        all_values.drop(drops, axis=1, inplace=True)

                hists = np.empty((len(self.data_components),
                                  len(all_values.columns),
                                  self.n_bins))
                binnings = []
                plotting_order_cols = []
                plotting_order_trans_cols = []
                weights = all_values.weight
                for j, c in enumerate(all_values.columns):
                    if c != 'weight':
                        df = all_values[[c, 'weight']]
                        df = df.dropna()
                        [obs_key, obs_key_t] = plot_labels[c]
                        plotting_order_cols.append(obs_key)
                        plotting_order_trans_cols.append(obs_key_t)
                        binning = binning_dict[c]
                        series = all_values[c]
                        for i, comp in enumerate(self.data_components):
                            hist = np.histogram(df[c][comp.name],
                                                bins=binning,
                                                weights=df['weight'][comp.name])[0]
                            if comp.ctype == 'MC':
                                    hists[i, j, :] = hist * self.data_livetime
                            else:
                                hists[i, j, :] = hist
                        binnings.append(binning)
                plotting_hists = aggregate(hists,
                                           self.cmds,
                                           list_data,
                                           list_plot)
                for i, comp in enumerate(self.plotting_components):
                    comp.hists = plotting_hists[i, :, :]
                    if comp.calc_uncertainties:
                        if len(self.alphas) > 0:
                            comp.uncertainties = calc_limits(comp.hists,
                                                             self.alphas)
                plotting_hists /= self.data_livetime
                plot_funcs.plot(outpath,
                                title,
                                self.plotting_components,
                                binnings,
                                plotting_order_cols,
                                plotting_order_trans_cols,
                                self.alphas,
                                self.plot_ratios)
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
        with open(os.path.join(outpath, 'observables.txt'), 'wb') as f:
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


def transform_values(transformation, values, key):
    if (transformation is None) or (transformation == 'None'):
        return values, key
    elif (transformation == 'log') or (transformation == 'log10'):
        values = np.log10(values)
        values.replace([np.inf, -np.inf], np.nan, inplace=True)
        return values, 'log10(%s)' % key
    elif (transformation == 'cos'):
        return np.cos(values), 'cos(%s)' % key
    elif (transformation == 'cosdeg'):
        return np.cos(np.deg2rad(values)), 'cos(%s)' % key
    elif (transformation == 'sin'):
        return np.sin(values), 'sin(%s)' % key
    elif (transformation == 'sindeg'):
        return np.sin(np.deg2rad(values)), 'sin(%s)' % key
    else:
        print('Invalid transformation \'%s\'' % transformation)
        return values, weights











