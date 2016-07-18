#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import warnings
import cPickle

def fields_view(arr, fields):
     dtype2 = np.dtype({name: arr.dtype.fields[name] for name in fields})
     return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)








def get_values_from_table(table, cols, dtype=float):
    values = np.empty(table.nrows,
                      dtype=[(k, type_)
                             for k, type_ in table.coldtypes.iteritems()])
    for i, row in enumerate(table.iterrows()):
        for key in cols:
            values[key][i] = row[key]
    return values


def filter_nans(values):
    for key in values.dtype.fields:
        values[key].mask = ~np.isfinite(values[key])
        exceptionally_big = np.where(
            np.abs(values[key][~values[key].mask]) > 1e50)[0]
        values[key].mask[exceptionally_big] = True
    return values


def filter_non_pos(values):
    for key in values.dtype.fields:
        values[key].mask = ~np.isfinite(values[key])
        below_zero = np.where(
            np.abs(values[key][~values[key].mask]) > <= 0)[0]
        values[key].mask[below_zero] = True
    return values


def transform_values(transformation, values, weights=None):
    if weights is None:
        if (transformation is None) or (transformation == 'None'):
            return values
        elif (transformation == 'log') or (transformation == 'log10'):
            values = filter_non_pos(values)
            return np.log10(values)
        elif (transformation == 'cos'):
            return np.cos(values)
        elif (transformation == 'cosdeg'):
            return np.cos(np.deg2rad(values))
        elif (transformation == 'sin'):
            return np.sin(values)
        elif (transformation == 'sindeg'):
            return np.sin(np.deg2rad(values))
        else:
            print('Invalid transformation \'%s\'' % transformation)
        return values
    else:
        if (transformation is None) or (transformation == 'None'):
            return values, weights
        elif (transformation == 'log') or (transformation == 'log10'):
            values, weights = filter_non_pos(values, weights)
            return np.log10(values), weights
        elif (transformation == 'cos'):
            return np.cos(values), weights
        elif (transformation == 'cosdeg'):
            return np.cos(np.deg2rad(values)), weights
        elif (transformation == 'sin'):
            return np.sin(values), weights
        elif (transformation == 'sindeg'):
            return np.sin(np.deg2rad(values)), weights
        else:
            print('Invalid transformation \'%s\'' % transformation)
            return values, weights


def transform_obs(transformation, observable):
    if (transformation is None) or (transformation == 'None'):
        return observable
    elif (transformation == 'log') or (transformation == 'log10'):
        return 'log10(%s)' % observable
    elif (transformation == 'cos'):
        return 'cos(%s)' % observable
    elif (transformation == 'cosdeg'):
        return 'cos(%s)' % observable
    elif (transformation == 'sin'):
        return 'sin(%s)' % observable
    elif (transformation == 'sindeg'):
        return 'sin(%s)' % observable
    else:
        print('Invalid transformation \'%s\'' % transformation)
        return observable
