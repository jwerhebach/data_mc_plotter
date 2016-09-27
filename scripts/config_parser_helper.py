#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function


def convert_list(obs_str):
    for s in [' ', '(', '[', '{', '}', ']', ')', '\t', '\n', "'", '"']:
        obs_str = obs_str.replace(s, '')
    return obs_str.split(',')

def split_obs_str(obs):
    if not isinstance(obs, list):
        obs = [obs]
    obs_dict = {}
    for o in obs:
        splitted = o.split('.')
        key = splitted[0]
        current_content = obs_dict.get(key, [[], []])
        try:
            current_content[0].append(splitted[1])
        except IndexError:
            print(o)
            exit()
        if len(splitted) == 3:
            current_content[1].append(splitted[2])
        else:
            current_content[1].append(None)
        obs_dict[key] = current_content
    return obs_dict
