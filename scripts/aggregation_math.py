#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from copy import deepcopy

OPS = ['*', '/', '+', '-']


def interpretate(s):
    i = 0
    while True:
        if i < len(s):
            s_i = s[i]
            for o in OPS:
                if (o in s_i) and len(s_i) > 1:
                    a, d, b = s_i.partition(o)
                    if d == o:
                        new_s = []
                        for j, s_j in enumerate(s):
                            if j == i:
                                new_s.append(a)
                                new_s.append(d)
                                new_s.append(b)
                            else:
                                new_s.append(s_j)
                        s = new_s
                        i += 1
                        break
            i += 1
        else:
            break
    return s


def interpretate_rec(s):
    if not isinstance(s, list):
        s = [s]
    s_new = interpretate(s)
    if s != s_new:
        return interpretate_rec(s_new)
    else:
        return s


def get_calc_steps(s):
    s = interpretate_rec(s)
    if len(s) == 1:
        return None
    steps = []
    counter = 0
    while True:
        for o in OPS:
            try:
                i = s.index(o)
            except ValueError:
                pass
            else:
                calculation = [s[i - 1], o, s[i + 1]]
                name = 'results_%d' % counter
                steps.append([name, calculation])
                counter += 1
                new_s = s[:i - 1]
                new_s.append(name)
                new_s.extend(s[i + 2:])
                s = new_s
                break
        if len(s) == 1:
            break
    return steps

def get_participants(cmd):
    splitted = interpretate_rec(cmd)
    for o in OPS:
        splitted = [s for s in splitted if s != o]
    return splitted


def aggregate(arr, cmds, list_data, list_plot):
    local_dict = {}
    for i, comp in enumerate(list_data):
        local_dict[comp] = arr[:, i, :]
    for i, cmd in enumerate(cmds):
        used_components = set()
        if any([cmd.startswith(o) for o in OPS]):
            raise ValueError('Aggregate cmd can not start with an operator')
        elif any([cmd.endswith(o) for o in OPS]):
            raise ValueError('Aggregate cmd can not end with an operator')
        else:
            steps = get_calc_steps(cmd)
        if steps is None:
            continue
        for result_name, calculation in steps:
            a = local_dict[calculation[0]]
            o = calculation[1]
            b = local_dict[calculation[2]]
            if o == '+':
                result = a + b
            elif o == '-':
                result = a - b
            elif o == '*':
                result = a * b
            elif o == '/':
                result = a / b
            else:
                raise ValueError('Invalid operator \'%s\'' % o)
            local_dict[result_name] = result
            used_components.add(result_name)
        local_dict[list_plot[i]] = result
        for k in used_components:
            del local_dict[k]
    plotting_hist = np.empty((arr.shape[0], len(list_plot), arr.shape[2]))
    for i, comp in enumerate(list_plot):
        plotting_hist[:, i, :] = local_dict[comp]
    return plotting_hist

if __name__ == '__main__':
    import numpy as np
    a = np.arange(100).reshape((10, 10))
    b = np.arange(100).reshape((10, 10))
    #b = b.T
    arr = np.zeros((10, 2, 10))
    arr[:, 0, :] = a
    arr[:, 1, :] = b
    cmds = ['a', 'a+b', 'a-b']

    list_data = ['a', 'b']
    list_plot = ['a', 'c', 'd']

    d = aggregate(arr,
                  cmds,
                  list_data,
                  list_plot)
    print(d)
