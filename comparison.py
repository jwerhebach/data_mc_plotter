#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os
import shutil

from ConfigParser import SafeConfigParser
from optparse import OptionParser

from scripts.component_handler import Component
import scripts.config_parser_helper as ch
from scripts.comparison_plotter import ComparisonPlotter


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-n", "--config_name",
                      dest="config_name",
                      type="str",
                      default=None)
    parser.add_option("-c", "--config_file",
                      dest="config_file",
                      type="str",
                      default=None)

    parser.add_option("-p", "--show_possible", action="store_true",
                      dest="possible",
                      default=False)

    parser.add_option("-f", "--force", action="store_true",
                      dest="force",
                      default=False)

    (opts, args) = parser.parse_args()
    if opts.config_name is not None:
        config_name = opts.config_name
        config_file = os.path.realpath('configs/%s.ini' % opts.config_name)
    elif opts.config_file is not None:
        config_file = os.path.realpath(opts.config_file)
        config_name = os.path.basename(config_file)
        config_name = config_name.replace('.ini', '')
    else:
        raise ValueError('Either option -c or -n is needed!')
    assert os.path.isfile(config_file), '%s not found'
    config = SafeConfigParser()
    l = config.read(config_file)

    outpath = None
    if config.has_option('General', 'Outpath'):
        outpath_opt = config.get('General', 'Outpath')
        if outpath_opt not in ['None', 'Default']:
            outpath_opt = os.path.realpath(outpath_opt)
            outpath = outpath_opt
    if outpath is None:
        outpath = os.path.realpath('output/%s' % config_name)
    if not opts.force:
        if os.path.isdir(outpath):
            i = 1
            while os.path.isdir(outpath + '_(%d)' % i):
                i += 1
            outpath = outpath + '_(%d)' % i
        os.makedirs(outpath)
    else:
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
    shutil.copy(config_file, outpath)


    components = []
    components_opts = ch.convert_list(config.get('General', 'Components'))
    for c in components_opts:
        assert config.has_section(c), \
            'For each component a section is neede\'%s\' missing!' % c
        component_dict = dict(config.items(c))
        curr_component = Component(c, component_dict)
        if curr_component not in components:
            components.append(curr_component)
        if curr_component.aggregation is not None:
            for i, c_a in enumerate(curr_component.aggregation.participant):
                if c_a not in components:
                    component_dict = dict(config.items(c_a))
                    components.append(Component(c_a, component_dict))
                index = components.index(c_a)
                if not curr_component.aggregation.keep_components:
                    components[index].show = False
                curr_component.aggregation.participant[i] = components[index]

    if config.has_option('General', 'Alphas'):
        alphas_ops = config.get('General', 'Alphas')
        alphas = [float(a) for a in ch.convert_list(alphas_ops)]
    else:
        alphas = [0.682689492, 0.9, 0.99]

    if config.has_option('General', 'IDKeys'):
        id_keys_opts = config.get('General', 'IDKeys')
        id_keys = ch.convert_list(id_keys_opts)
    else:
        id_keys = []

    obs = '*'
    if config.has_option('General', 'Observables'):
        obs_opts = config.get('General', 'Observables')
        if obs_opts not in ['all', '*']:
            obs = ch.convert_list(obs_opts)


    comp_plotter = ComparisonPlotter(components,
                                     id_keys,
                                     match=False,
                                     n_bins=50)

    if config.has_option('General', 'Title'):
        title = config.get('General', 'Title')
    else:
        title = ''

    if opts.possible or obs == '*':
        blacklist = {'obs': [],
                     'tabs': [],
                     'cols': []}

        if config.has_section('Blacklist'):
            if config.has_option('Blacklist', 'Observables'):
                blacklist_obs_opts = config.get('Blacklist',
                                                'Observables')
                if blacklist_obs_opts != ['None', 'none']:
                    blacklist['obs'].extend(
                        ch.convert_list(blacklist_obs_opts))
            if config.has_option('Blacklist', 'Tables'):
                blacklist_tabs_opts = config.get('Blacklist',
                                                 'Tables')
                if blacklist_tabs_opts != ['None', 'none']:
                    blacklist['tabs'].extend(
                        ch.convert_list(blacklist_tabs_opts))
            if config.has_option('Blacklist', 'Columns'):
                blacklist_tabs_opts = config.get('Blacklist',
                                                 'Columns')
                if blacklist_tabs_opts != ['None', 'none']:
                    blacklist['cols'].extend(
                        ch.convert_list(blacklist_tabs_opts))
        obs = comp_plotter.get_possiblites(outpath=outpath,
                                           blacklist=blacklist)
        if opts.possible:
            sys.exit()


    comp_plotter.plot(title=title,
                      observables=obs,
                      outpath=outpath,
                      alphas=alphas)
