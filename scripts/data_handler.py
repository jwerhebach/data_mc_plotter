#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import warnings
import numpy as np
import tables

def get_values_from_table(table, cols, dtype=float):
    values = np.empty((table.nrows, len(cols)), dtype=float)
    for i, row in enumerate(table.iterrows()):
        values[i, :] = [row[col] for col in cols]
    return values

def filter_nans(values, weights=None):
    nan_mask = ~np.isnan(values)
    finite_mask = np.isfinite(values)
    filter_mask = np.logical_and(nan_mask, finite_mask)
    if weights is None:
        return values[filter_mask], None
    else:
        weights = weights[filter_mask]
        values = values[filter_mask]
        return values, weights.reshape(values.shape)

def transform_values(values, transformation):
    if (transformation is None) or (transformation == 'None'):
        return values
    elif (transformation == 'log') or (transformation == 'log10'):
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

def transform_obs(observable, transformation):
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
