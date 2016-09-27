#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import glob
import warnings
import numpy as np
from numpy.lib.recfunctions import stack_arrays
import pandas as pd
import tables

from .plot_funcs import get_color, get_cmap
from .aggregation_math import get_participants
from .config_parser_helper import split_obs_str, convert_list


class Component:
    def __init__(self,
                 name,
                 option_dict):
        self.name = name
        self.ctype = option_dict['type']
        self.label = option_dict['label']
        self.color = option_dict.get('color', None)
        self.id_cols = ['Component']
        directory = option_dict.get('directory', None)
        file_list = option_dict.get('filelist', None)
        max_files = int(option_dict.get('maxfiles', -1))
        aggregation = option_dict.get('aggregation', None)
        if aggregation is None:
            self.aggregation = None
            assert (directory is not None) or (file_list is not None), \
                'If component is not from aggregation directory or file_list'\
                'is needed!'
            if file_list is None:
                if '*' not in directory:
                    directory += '*'
                file_list = glob.glob(directory)
            else:
                file_list = convert_list(file_list)
            if max_files > 0:
                upto = max(len(file_list), int(max_files))
            self.livetime = float(option_dict.get('livetime', 1.))
            self.weight = option_dict.get('weight', None)
            if self.weight in ['none', 'None', 'NONE']:
                self.weight = None
            self.file_list = file_list
            self.scaling_factor = float(option_dict.get('scale', 1.))
        else:
            keep_components = option_dict.get('keepcomponents', False)
            if isinstance(keep_components, str):
                if keep_components in ['True', 'true', 'TRUE']:
                    keep_components = True
                else:
                    keep_components = False
            self.aggregation = self.Aggregation(aggregation,
                                                keep_components)
            self.file_list = None
            self.livetime = None
            self.weight = None

        if self.color is None:
            self.color = get_color()

        self.show = True

        self.hists = None
        self.uncertainties = None
        self.uncert_ratio = None
        self.calc_uncertainties = option_dict.get('showuncertainties', False)
        if self.calc_uncertainties:
            self.cmap = get_cmap()

    def init_component(self, id_table, id_cols):
        self.id_table = id_table
        self.id_cols.extend(id_cols)
        if self.aggregation is None:
            if self.weight is not None:
                weight_dict = split_obs_str(self.weight)
                weight_table = weight_dict.keys()[0]
                cols = weight_dict[weight_table][0]
                self.weight = self.get_values(weight_table,
                                              cols,
                                              add_weight=False)
                self.weight.rename(columns={cols[0]: 'weight'}, inplace=True)
            else:
                self.weight = self.get_values(id_table,
                                              id_cols,
                                              add_weight=False)
                self.weight['weight'] = 1.
            if self.ctype == 'MC':
                self.weight /= self.livetime
            self.weight *= self.scaling_factor
            self.scaling_factor = 1.

    def get_values(self, table_key, cols, add_weight=True):
        if isinstance(cols, str):
            cols = [cols]
        for i, file_name in enumerate(self.file_list):
            f = pd.HDFStore(file_name, 'r')
            table = f[table_key]
            drops = [c for c in table.columns
                     if c not in cols and c not in self.id_cols]
            table.drop(drops, axis=1, inplace=True)
            table['Component'] = self.name
            table.set_index(self.id_cols, inplace=True)
            f.close()
            if i == 0:
                values = table
            else:
                values = values.append(table)
        if add_weight:
            if isinstance(self.weight, float):
                values['weight'] = self.weight
            else:
                values = values.join(self.weight)
        return values

    def get_observables(self, blacklist, check_all=False):
        [bl_tabs, bl_cols, bl_obs] = blacklist
        component_set = set()
        if check_all:
            files = self.file_list
        else:
            files = self.file_list[:1]
        for i, file_name in enumerate(files):
            file_set = set()
            f = tables.open_file(file_name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for table in f.iter_nodes('/', classname='Table'):
                    table_name = table.name
                    if table_name not in bl_tabs:
                        col_names = [c for c in table.colnames
                                     if c not in bl_cols]
                        for c in col_names:
                            obs = '%s.%s' % (table_name, c)
                            if obs not in bl_obs:
                                file_set.add(obs)
                f.close()
                if i == 0:
                    component_set = file_set
                else:
                    component_set = component_set.intersection(file_set)
        return component_set

    def __eq__(self, b):
        if isinstance(b, str):
            return self.name == b
        elif isinstance(b, Component):
            return self.name == b.name
        else:
            return False

    def __cmp__(self, b):
        return self.__eq__(b)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self.name))

    def __repr__(self):
        return '%s (%s)' % (self.name, self.ctype)

    def get_nevents(self, weighted=False):
        if weighted:
            if self.aggregation is not None:
                n_events = 0.
                for c in self.aggregation.participant:
                    n_events += c.get_nevents(weighted=True)
                return n_events
            else:
                return self.weight.weight.sum()
        else:
            if self.aggregation is not None:
                n_events = 0
                for c in self.aggregation.participant:
                    n_events += c.get_nevents()
                return n_events
            else:
                return len(self.weight)

    def set_scaling(self, scaling_factor):
        if self.aggregation is None:
            self.scaling_factor *= scaling_factor
            if self.weight is not None:
                self.weight *= self.scaling_factor
                self.scaling_factor = 1.
                self.nevents_weighted = np.sum(self.weight)
            elif self.get_nevents() is not None:
                self.weight = np.ones(self.get_nevents()) * \
                    self.scaling_factor
                self.scaling_factor = 1.
                self.nevents_weighted = np.sum(self.weight)

        else:
            for c in self.aggregation.participant:
                c.set_scaling(scaling_factor)


    class Aggregation:
        def __init__(self, cmd, keep_components):
            self.cmd = cmd
            self.participant = get_participants(cmd)
            self.keep_components = keep_components
