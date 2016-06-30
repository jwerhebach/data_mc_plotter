#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function


def convert_list(obs_str):
    for s in [' ', '(', '[', '{', '}', ']', ')', '\t', '\n', "'", '"']:
        obs_str = obs_str.replace(s, '')
    return obs_str.split(',')


def check_for_complete_config(config):
    assert config.has_section('General'), \
        "Section 'General' with options 'Level' and Components is mandatory"
    assert config.has_option('General', 'Level'), \
        "Option 'Level' in Section General is mandatory"
    assert config.has_option('General', 'Components'), \
        "Option 'Components' in Section General is mandatory"
