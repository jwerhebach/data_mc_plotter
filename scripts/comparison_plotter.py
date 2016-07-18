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
        self.id_dict = ch.split_obs_str(id_keys)
        assert len(self.id_dict.keys()) == 1, \
            'id_keys have to be from one table!'
        self.id_cols = self.id_dict[self.id_dict.keys()[0]][0]
        self.match = match
        self.autotransform = autotransform
        self.n_bins = n_bins
        self.alphas = sorted(alphas)

        self.data_livetime = 0.
        self.n_events_data = 0

        for i, c in enumerate(components):
            c.init_component(self.id_dict)
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
        hists = np.zeros((len(self.data_components), n_obs, self.n_bins))
        binnings = np.zeros((n_obs, self.n_bins + 1))
        transformed_obs_keys = []
        obs_keys = []
        cols_mask = np.ones(n_obs, dtype=bool)
        finished_cols = 0
        with tqdm(total=n_obs, unit='Observables') as pbar:
            for table_key, [cols, trafos] in observables.iteritems():
                all_values = {}
                limits = np.ones((len(self.data_components),
                                  len(cols),
                                  2))
                limits[:, :, 0] = np.inf
                limits[:, :, 1] = -np.inf
                quantiles = np.ones((len(self.data_components),
                                  len(cols),
                                  len(percentiles)))
                for i, comp in enumerate(self.data_components):
                    all_values[comp.name] = {}
                    comp_values = comp.get_values(table_key, cols)
                    comp_values = dh.filter_nans(comp_values)
                    all_values[comp.name] = comp_values
                    for j, [col, trafo] in enumerate(zip(cols, trafos)):
                        current_col = comp_values[col]
                        if cols_mask[finished_cols + j]:
                            if np.sum(current_col.mask) == len(current_col):
                                all_values[comp.name][col] = {}
                                col_dict = all_values[comp.name][col]
                                vals = comp_values[:, j][filter_mask]
                                col_dict['values'] = vals
                                col_dict['filter_mask'] = filter_mask
                                limits[i, j, 0] = np.min(vals)
                                limits[i, j, 1] = np.max(vals)
                                if limits[i, j, 0] == limits[i, j, 1]:
                                    cols_mask[finished_cols + j] = False
                                if self.autotransform:
                                    quantiles[i, j, :] = np.percentile(
                                        vals,
                                        percentiles)
                                elif trafo == 'log' and limits[i, j, 0] <= 0.:
                                    limits[i, j, 0] = np.min(vals[vals > 0.])
                            else:
                                cols_mask[finished_cols + j] = False
                min_vals = np.min(limits[:, :, 0], axis=0)
                max_vals = np.min(limits[:, :, 1], axis=0)
                for j, [col, trafo] in enumerate(zip(cols, trafos)):
                    current_col = finished_cols + j
                    min_val = min_vals[j]
                    max_val = max_vals[j]
                    obs_key = '%s.%s' % (table_key, col)
                    if cols_mask[current_col]:
                        diff = max_val - min_val
                        if self.autotransform:
                            if trafo in ['None', None]:
                                quantiles_all = np.average(
                                    quantiles[:, current_col, :]
                                    weights=n_events,
                                    axis=0)
                            if any(quantiles_all)
                            median = quantiles_all[2]
                            ratio_media = (median - min_val) / diff
                            if ((ratio_media < 0.1) and min_val > 0.0):
                                trafo = 'log10'
                            elif np.absolute(diff - np.pi) < 1e-3:
                                trafo = 'cos'
                            else:
                                trafo = None
                        transformed_obs_key = dh.transform_obs(trafo, obs_key)
                        obs_keys.append(obs_key)
                        transformed_obs_keys.append(transformed_obs_key)
                        transformed_limits = dh.transform_values(
                            trafo,
                            np.asarray([min_val, max_val]))
                        min_val = transformed_limits[0]
                        max_val = transformed_limits[1]
                        if max_val < min_val:
                            max_val, min_val = min_val, max_val
                        diff = max_val - min_val
                        offset = diff * 1e-5
                        binnings[current_col] = np.linspace(
                            min_val - offset,
                            max_val + offset,
                            self.n_bins + 1)
                        for i, comp in enumerate(self.data_components):
                            col_dict = all_values[comp.name][col]
                            weights = comp.weight[col_dict['filter_mask']]
                            vals, weights = dh.transform_values(
                                trafo, col_dict['values'], weights)
                            weights = weights.reshape(vals.shape)
                            hist = np.histogram(vals,
                                                bins=binnings[current_col],
                                                weights=weights)[0]
                            if comp.ctype == 'MC':
                                hists[i, current_col, :] = hist * self.data_livetime
                            else:
                                hists[i, current_col, :] = hist
                finished_cols += len(cols)
                pbar.update(len(cols))

        hists = hists[:, np.where(cols_mask)[0], :]
        plotting_hists = aggregate(hists,
                                   self.cmds,
                                   list_data,
                                   list_plot)
        binnings = binnings[np.where(cols_mask)[0]]
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
                        obs_keys,
                        transformed_obs_keys,
                        self.alphas,
                        self.plot_ratios)

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
