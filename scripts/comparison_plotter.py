#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os
import warnings

from tqdm import tqdm

import numpy as np
import tables

from matplotlib import pyplot as plt

from .aggregation_math import aggregate
from .aggarwal_err import calc_limits
import plot_funcs


class ComparisonPlotter:
    def __init__(self,
                 system_config,
                 components,
                 id_keys,
                 aggregations=None,
                 sum_mc=True,
                 match=False,
                 n_bins=50):
        self.sys_conf = system_config
        self.components = components
        self.id_dict = self.__split_obs_str__(id_keys)
        assert len(self.id_dict.keys()) == 1, \
            'id_keys have to be from one table!'
        self.id_obs = self.id_dict[self.id_dict.keys()[0]][0]
        self.match = match
        self.n_bins = n_bins
        self.livetime = 0.
        self.cc = 0
        if self.aggregations is not None:
            self.aggregations = aggregations
            self.aggregated_component = []
            for k in aggregations.keys():
                color =
                self.aggregated_components.append({'dataset': k,
                                                   'type': 'MC',})
        else:
            self.aggregations = {}
            self.aggregated_components = []
        if sum_mc:
            mc_comps = []
            for c in self.components:
                if c['type'] == 'MC':
                    mc_comps.append(c['dataset'])
            if len(mc_comps) == 0:
                raise ValueError('\'SumMC\' choosen without MC components')
            else:
                cmd = mc_comps[0]
                for c_i in mc_comps[1:]:
                    cmd += '+%s' % c_i
                self.aggregated_components({'dataset': 'Sum',
                                            'type': MC})
                self.aggregations['Sum'] = cmd

    def plot(self, observables, outpath, uncertainties=''):
        observables = self.__split_obs_str__(observables)
        for i, c in enumerate(self.components):
            self.components[i] = self.__init_component__(c)
            if 'color' not in self.components[i].keys():
                self.components[i]['color'] = plot_funcs.colors[self.cc]
                self.cc += 1
        print('Fetching Observable infos')
        n_obs = np.sum([len(observables[k][0]) for k in observables.keys()])
        with tqdm(total=n_obs, unit='Observables') as pbar:
            for table_key, [cols, transformations] in observables.iteritems():
                informations = {}
                trans_obs = {}
                binning_dict = {}
                for i, c in enumerate(self.components):
                    vals = self.__get_values_from_compontent__(c,
                                                               table_key,
                                                               cols)
                    for j, [col, t] in enumerate(zip(cols, transformations)):
                        obs_key = '%s.%s' % (table_key, col)
                        filtered_vals, filtered_weights = self.__filter__(
                            vals[:, j],
                            c['weights'])
                        filtered_vals, trans_o = self.__transform_values__(
                            filtered_vals, t, obs_key)

                        if len(filtered_vals) != 0:
                            if i == 0:
                                max_val = np.max(filtered_vals)
                                min_val = np.min(filtered_vals)
                                binning = np.linspace(min_val, max_val,
                                                      self.n_bins + 1)
                                informations[obs_key] = {}
                                binning_dict[obs_key] = binning
                                trans_obs[obs_key] = trans_o
                                if max_val == min_val:
                                    break
                            hist = np.histogram(filtered_vals,
                                                bins=binning,
                                                weights=filtered_weights)[0]
                            hist *= self.livetime
                            informations[obs_key][c['dataset']] = hist
                for obs_key in informations.keys():
                    trans_o = trans_obs[obs_key]
                    binning = binning_dict[obs_key]
                    col_dict = informations[obs_key]

                    plot_keys = set(col_dict.keys())
                    for key, cmd in self.aggregations.iteritems():
                        if key == 'Sum':
                            col_dict, _ = aggregate(col_dict, cmd, key)
                        else:
                            col_dict, used_keys = aggregate(col_dict, cmd, key)
                            plot_keys = plot_keys.difference(used_keys)
                    for key in plot_keys:
                        hist = col_dict[key]
                        if key == uncertainties:
                            err_y = calc_limits(hist)
                            err_y[0] = hist - err_y[0]
                            err_y[1] = err_y[1] - hist
                        else:
                            err_y = None
                        bin_center = np.diff(binning)
                        x_err_left = bin_center - binning[:-1]
                        x_err_right = binning[1:] - bin_center
                        f, ax = plt.subplots()
                        ax.hist(bin_center,
                                bins=binning,
                                weights=hist,
                                log=True,
                                histtype='step',
                                color = )

pyplot.hist(x, bins=10, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None, **kwargs)




                pbar.update(len(cols))

    def __split_obs_str__(self, obs, return_dict=True):
        if not isinstance(obs, list):
            obs = [obs]
        obs_dict = {}
        for o in obs:
            splitted = o.split('.')
            key = splitted[0]
            current_content = obs_dict.get(key, [[], []])
            current_content[0].append(splitted[1])
            if len(splitted) == 3:
                current_content[1].append(splitted[2])
            else:
                current_content[1].append(None)
            obs_dict[key] = current_content
        return obs_dict

    def __transform_values__(self, values, transformation, observable):
        if (transformation is None) or (transformation == 'None'):
            return values, observable
        elif (transformation == 'log') or (transformation == 'log10'):
            return np.log10(values), 'log10(%s)' % observable
        elif (transformation == 'cos'):
            return np.cos(values), 'cos(%s)' % observable
        elif (transformation == 'cosdeg'):
            return np.cos(np.deg2rad(values)), 'cos(%s)' % observable
        elif (transformation == 'sin'):
            return np.sin(values), 'sin(%s)' % observable
        elif (transformation == 'sindeg'):
            return np.sin(np.deg2rad(values)), 'sin(%s)' % observable
        else:
            print('Invalid transformation \'%s\'' % transformation)
            return values, observable

    def __get_ids__(self, component):
        files = component['files']
        for i, file_name in enumerate(files):
            f = tables.open_file(file_name)
            table_key = self.id_dict.keys()[0]
            table = f.get_node('/%s' % table_key)
            [cols, _] = self.id_dict[table_key]
            id_array = self.__get_values_from_table__(table,
                                                      cols,
                                                      dtype=int)
            f.close()
            if i == 0:
                full_id_array = id_array
            else:
                full_id_array = np.vstack((full_id_array, id_array))
        mag = np.asarray(np.log10(np.max(full_id_array, axis=0) + 1) + 1,
                         dtype=int)
        mag = np.cumsum(mag[::-1])[::-1]
        mag[:-1] = mag[1:]
        mag[-1] = 0
        mag = np.power(10, mag)

        def id_func(arr):
            return np.sum(arr * mag, axis=1, dtype=int)

        def id_func_rev(ids):
            divided = np.asarray([ids for i in range(len(mag))]).T
            divided = divided // mag * mag
            divided[:, 1:] = np.diff(divided, axis=1)
            divided = divided / mag
            return divided

        ids = id_func(full_id_array)
        return ids, id_func, id_func_rev

    def __init_component__(self, component):
        ids, id_func, id_func_r = self.__get_ids__(component)
        component['ids'] = ids
        component['id_func'] = id_func
        component['id_func'] = id_func_r
        component['n_events'] = len(ids)
        if len(ids) != len(np.unique(ids)):
            print('IDs not unique for \'%s\'!' % component['dataset'])
        if component['type'] == 'MC':
            if component['weight'] in ['none', 'None']:
                component['weight'] = np.ones(component['n_events'],
                                              dtype=float)
            else:
                weight_dict = self.__split_obs_str__(component['weight'])
                weight_table = weight_dict.keys()[0]
                cols = weight_dict[weight_table][0]
                component['weight'] = self.__get_values_from_compontent__(
                    component,
                    weight_table,
                    cols)
            component['weight'] /= component['n_files']
        elif component['type'] == 'Data':
            self.livetime += component['livetime']
            component['weight'] = np.ones(component['n_events'])
            component['color'] = 'k'
        return component

    def __filter__(self, values, weights=None):
        nan_mask = ~np.isnan(values)
        finite_mask = np.isfinite(values)
        filter_mask = np.logical_and(nan_mask, finite_mask)
        if weights is None:
            return values[filter_mask]
        else:
            return values[filter_mask], weights[filter_mask]

    def __get_values_from_compontent__(self, component, table_key, cols):
        files = component['files']
        n_events = component['n_events']
        from_i = 0
        values = np.empty((n_events, len(cols)))
        for i, file_name in enumerate(files):
            f = tables.open_file(file_name)
            table = f.get_node('/%s' % table_key)
            values_f = self.__get_values_from_table__(table,
                                                      cols,
                                                      dtype=int)
            f.close()
            events_in_file = values_f.shape[0]
            to_i = from_i + events_in_file
            values[from_i:to_i, :] = values_f
            from_i += events_in_file
        if from_i != n_events:
            difference = n_events - from_i
            print('\'%s\' from \'%s\' is missing %d Events' %
                  (table, component['dataset'], difference))
        return values

    def __get_values_from_table__(self, table, cols, dtype=float):
        values = np.empty((table.nrows, len(cols)), dtype=float)
        for i, row in enumerate(table.iterrows()):
            values[i, :] = [row[col] for col in cols]
        return values

    def get_possiblites(self,
                        outpath,
                        check_all=False,
                        blacklist=None):
        if isinstance(blacklist, dict):
            blacklist_cols = blacklist['cols']
            blacklist_cols.extend(self.id_obs)
            blacklist_obs = blacklist['obs']
            blacklist_tabs = blacklist['tabs']
        else:
            blacklist_cols = self.id_obs
        blacklist_obs = set(blacklist_obs)
        for i, c in enumerate(self.components):
            files = c['files']
            for file_name in files:
                f = tables.open_file(file_name)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    component_set = set()
                    for table in f.iter_nodes('/', classname='Table'):
                        table_name = table.name
                        if table_name not in blacklist_tabs:
                            col_names = table.colnames
                            for id_obs in blacklist_cols:
                                if id_obs in col_names:
                                    col_names.remove(id_obs)
                            for c in col_names:
                                obs = '%s.%s' % (table_name, c)
                                if obs not in blacklist_obs:
                                    component_set.add(obs)
                    f.close()
                    if i == 0:
                        obs_set = component_set
                    else:
                        obs_set = obs_set.intersection(component_set)
                    if not check_all:
                        break
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
