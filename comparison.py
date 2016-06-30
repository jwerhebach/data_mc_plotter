#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import os
import shutil

from ConfigParser import SafeConfigParser
from optparse import OptionParser

from scripts.comparison_plotter import ComparisonPlotter
from scripts.file_grabber import FileGrabber
from scripts.config_parser_helper import convert_list


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

    assert config.has_section('General'), \
        "Section 'General' with options 'Level' and Components is mandatory"
    assert config.has_option('General', 'Level'), \
        "Option 'Level' in Section General is mandatory"
    assert config.has_option('General', 'Components'), \
        "Option 'Components' in Section General is mandatory"

    level = config.getint('General', 'Level')

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

    file_grabber = FileGrabber(system_config=syst_conf,
                               level=level)
    components = convert_list(config.get('General', 'Components'))
    for c in components:
        assert config.has_section(c), \
            "For each component a section is needed"
        has_needed_opts = config.has_option(c, 'Name') and \
            config.has_option(c, 'Type')
        component_type = config.get(c, 'Type')
        if component_type == 'MC':
            has_needed_opts = has_needed_opts and \
                config.has_option(c, 'Weight')
            assert has_needed_opts, \
                '\'MC\' component needs \'Name\', and \'Weight\''
            dataset_name = config.get(c, 'Name')
            if config.has_option(c, 'MaxFiles'):
                max_files = config.getint(c, 'MaxFiles')
            else:
                max_files = -1
            file_grabber.add_component(
                {'dataset': config.get(c, 'Name'),
                 'weight': config.get(c, 'Weight'),
                 'type': config.get(c, 'Type'),
                 'max_files': max_files})
        elif component_type == 'Data':
            has_needed_opts = has_needed_opts and \
                config.has_option(c, 'Burnsample')
            assert has_needed_opts, \
                '\'Data\' component needs \'Name\', and \'Burnsample\''
            if config.has_option(c, 'MaxFiles'):
                max_files = config.getint(c, 'MaxFiles')
            else:
                max_files = -1
            file_grabber.add_component(
                {'dataset': config.get(c, 'Name'),
                 'burnsample': config.getboolean(c, 'Burnsample'),
                 'type': config.get(c, 'Type'),
                 'max_files': max_files})
        else:
            print(component_type)
            raise ValueError('Component Type can be either \'MC\' or \'Data\'')

    components = file_grabber.get_component_dict()

    if config.has_option('General', 'IDKeys'):
        id_keys_opts = config.get('General', 'IDKeys')
        id_keys = convert_list(id_keys_opts)
    else:
        id_keys = []

    if config.has_option('General', 'SumMC'):
        sum_mc = config.getboolean('General', 'SumMC')
    else:
        sum_mc = False

    if config.has_section('Aggregations'):
        aggregations = dict(config.items('Aggregations'))
    else:
        aggregations = None

    comp_plotter = ComparisonPlotter(system_config=syst_conf,
                                     components=components,
                                     id_keys=id_keys,
                                     sum_mc=sum_mc,
                                     aggregations=aggregations)

    obs = '*'
    if config.has_option('General', 'Observables'):
        obs_opts = config.get('General', 'Observables')
        if obs_opts not in ['all', '*']:
            obs = convert_list(obs_opts)

    if config.has_option('General', 'Uncertainties'):
        uncer_opts = config.get('General', 'Uncertainties')
        if uncer_opts in ['mc', 'MC', 'Mc']:
            uncertainties = 'Sum'
        elif uncer_opts is None or uncer_opts in ['None', 'none']:
            uncertainties = ''
        else:
            uncertainties = uncer_opts
    else:
        uncertainties = 'Sum'

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
                        convert_list(blacklist_obs_opts))
            if config.has_option('Blacklist', 'Tables'):
                blacklist_tabs_opts = config.get('Blacklist',
                                                 'Tables')
                if blacklist_tabs_opts != ['None', 'none']:
                    blacklist['tabs'].extend(
                        convert_list(blacklist_tabs_opts))
            if config.has_option('Blacklist', 'Columns'):
                blacklist_tabs_opts = config.get('Blacklist',
                                                 'Columns')
                if blacklist_tabs_opts != ['None', 'none']:
                    blacklist['cols'].extend(
                        convert_list(blacklist_tabs_opts))
        obs = comp_plotter.get_possiblites(outpath=outpath,
                                           blacklist=blacklist)
        if opts.possible:
            sys.exit()
    comp_plotter.plot(observables=obs,
                      outpath=outpath,
                      uncertainties=uncertainties)
