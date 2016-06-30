#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

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


def aggregate(arr_dict, cmd, key, keep=False):
    used_components = set()
    local_dict = deepcopy(arr_dict)
    if any([cmd.startswith(o) for o in OPS]):
        raise ValueError('Aggregate cmd can not start with an operator')
    elif any([cmd.endswith(o) for o in OPS]):
        raise ValueError('Aggregate cmd can not end with an operator')
    else:
        steps = get_calc_steps(cmd)
    for result_name, calculation in steps:
        a = local_dict[calculation[0]]
        used_components.add(calculation[0])
        o = calculation[1]
        b = local_dict[calculation[2]]
        used_components.add(calculation[2])
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
    local_dict[key] = result
    for k in arr_dict.keys():
        used_components.discard(k)
    for k in used_components:
        del local_dict[k]
    return local_dict, used_components

if __name__ == '__main__':
    import numpy as np
    a = np.arange(10)
    b = np.arange(10)[::-1]
    cmd = 'a+b'
    d = aggregate({'a': a,
                   'b': b,
                   'd': a},
                  cmd,
                  'c')
    print(d)
