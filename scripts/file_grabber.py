#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os
import re
from fnmatch import fnmatch

from multiyear_processing.modules.simulation_data import DatasetHandler
from multiyear_processing.modules.experimental_data import DetectorHandler


class FileGrabber:
    def __init__(self, system_config, level, match_keys=None):
        self.sys_conf = system_config
        self.sim_handler = None
        self.exp_handler = None
        self.level = level
        self.components = []
        self.match_keys = match_keys

    def add_component(self, component):
        if component['type'] == 'MC':
            name = component['dataset']
            max_files = component['max_files']
            files, n_files = self.__fetch_files_sim__(name,
                                                      max_files)
            component['files'] = files
            component['n_files'] = n_files
            if n_files > 0:
                self.components.append(component)
            else:
                print("Warning: Ignored '%s' because no files were found!"
                      % name)
        elif component['type'] == 'Data':
            name = component['dataset']
            burnsample = component['burnsample']
            max_files = component['max_files']
            files, livetime = self.__fetch_files_exp__(name,
                                                       burnsample,
                                                       max_files)
            component['files'] = files
            component['livetime'] = livetime
            if livetime > 0.:
                self.components.append(component)
            else:
                print("Warning: Ignored '%s' because no files were found!"
                      % name)

    def get_component_dict(self):
        return self.components

    def __fetch_files_sim__(self, dataset_name, max_files=-1):
        if self.sim_handler is None:
            self.sim_handler = DatasetHandler(self.sys_conf)
        data_info_card = self.sim_handler.get_dataset(dataset_name)
        local_path = data_info_card['local_path']
        local_path = os.path.join(local_path, 'level%d_hdf5' % self.level)
        filename_pattern = data_info_card['filename_pattern']
        filename_pattern = filename_pattern.replace('$LEVEL', str(self.level))
        filename_pattern = filename_pattern.replace('$FILENUM', '*')
        filename_pattern = filename_pattern.replace('.i3.bz2', '.hd5')
        in_file_list = []
        n_files = 0
        for path, subdirs, files in os.walk(local_path):
            for filename in files:
                if fnmatch(filename, filename_pattern):
                    f = os.path.join(path, filename)
                    current_files = re.findall('\d+Files', filename)[-1]
                    n_files += int(current_files.replace('Files', ''))
                    in_file_list.append(str(f))
                    if len(in_file_list) == max_files:
                        break
        return in_file_list, n_files

    def __fetch_files_exp__(self,
                            detector_name,
                            burnsample=True,
                            max_files=-1):
        if self.exp_handler is None:
            self.exp_handler = DetectorHandler(system_config=self.sys_conf)
        self.exp_handler.switch_detector(detector=detector_name,
                                         burnsample=burnsample)
        detector = self.exp_handler.detector
        runs, n_runs = self.exp_handler.get_detector_runs()
        filename_pattern = detector['filename_pattern']
        filename_pattern = filename_pattern.replace('$LEVEL', str(self.level))
        in_file_list = []
        livetime = 0.
        for run in runs:
            local_path = self.sys_conf.generate_local_path(run.local_path)
            local_path = os.path.join(local_path, 'level%d_hdf5' % self.level)
            run_string = 'Run%8d' % (run.run_id)
            run_string = run_string.replace(' ', '0')
            run_file_name = filename_pattern.replace('$RUN_$SUBRUN.i3.bz2',
                                                     '%s.hd5' % run_string)
            file_path = os.path.join(local_path, run_file_name)
            if os.path.isfile(file_path):
                in_file_list.append(file_path)
                livetime += run.livetime
                if len(in_file_list) == max_files:
                    break
        return in_file_list, livetime
