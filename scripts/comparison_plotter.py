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
                 autotransform=False,
                 n_bins=50,
                 alphas=[0.682689492, 0.9, 0.99]):
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

    def auto_scale(self, scaling_list):
        for c_i in scaling_list:
            index = self.components.index(c_i)
            curr_component = self.components[index]
            sum_w_i = curr_component.get_nevents(weighted=True)
            n_events_i = sum_w_i * self.data_livetime
            scaling_factor = self.n_events_data/n_events_i
            curr_component.set_scaling(scaling_factor)
            print('Scaling for \'%s\' set to: %.4f' % (c_i, scaling_factor))

    def fetch_data_and_plot(self, title, observables, outpath):
        observables = ch.split_obs_str(observables)
        print('Fetching Observable infos')
        n_obs = np.sum([len(observables[k][0]) for k in observables.keys()])
        list_data = [str(c) for c in self.data_components]
        list_plot = [str(c) for c in self.plotting_components]
        hists = np.zeros((len(self.data_components), n_obs, self.n_bins))
        binnings = np.zeros((n_obs, self.n_bins + 1))
        trans_obs_keys = []
        obs_keys = []
        cols_mask = np.ones(n_obs, dtype=bool)
        finished_cols = 0
        with tqdm(total=n_obs, unit='Observables') as pbar:
            for table_key, [cols, trans] in observables.iteritems():
                all_values = []
                for comp in self.data_components:
                    all_values.append(comp.get_values(table_key, cols))
                for j, [col, trafo] in enumerate(zip(cols, trans)):
                    val = all_values[i][: ,]
                    mean_all = 0.
                    sum_w_all = 0.
                    min_vals = []
                    max_vals = []
                    for i, comp in enumerate(self.data_components):
                        val = all_values[i][:, j] =

                        min_j, max_j, mean = self.get_stat_infos
                        min_vals.append(min_j)
                        max_vals.append(max_j)
                        sum_w_j = self.comp.get_nevents(weighted=True)
                        mean_all += mean*sum_w_j
                        sum_w_all += sum_w_j


                    min_all = min(min_vals)
                    max_all = max(max_vals)
                    mean_all /= sum_wall
                    if self.autotransform:
                        trafo = None


                    obs_key = '%s.%s' % (table_key, c)
                    trans_obs_keys.append(dh.transform_obs(obs_key, trafo))
                    obs_keys.append(obs_key)
                    [min_all, max_all] = dh.transform_values(trafo,
                                                             [min_all,
                                                              max_all])
                    vals, weights = dh.transform_values(t, all_values[j],
                                                           weights)

                finished_cols += len(cols)
                pbar.update(len(cols))








                if True:
                    comp_values = comp.get_values(table_key, cols)
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
                                hist = np.histogram(vals,
                                                    bins=binnings[j],
                                                    weights=weights)[0]
                                if comp.ctype == 'MC':
                                    hists[i, current_col, :] = hist * self.livetime
                                else:
                                    hists[i, current_col, :] = hist
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
                    comp.uncertainties = calc_limits(comp.hists,
                                                     self.alphas)
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
