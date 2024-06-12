#!/usr/bin/env python

# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

import onelib.utils as oneutils


def _simple_warning(message, category, filename, lineno, file=None, line=None):
    return f'{category.__name__}: {message}\n'


class CfgRunner:
    driver_sequence = [
        'one-optimize', 'one-quantize', 'one-pack', 'one-codegen', 'one-profile',
        'one-partition', 'one-infer'
    ]

    def __init__(self, path):
        self.path = path
        self.optparser = None
        self.cfgparser = oneutils.get_config_parser()
        parsed = self.cfgparser.read(os.path.expanduser(path))
        if not parsed:
            raise FileNotFoundError('Not found given configuration file')

        self._verify_cfg(self.cfgparser)
        # default import drivers
        self.import_drivers = [
            'one-import-bcq', 'one-import-onnx', 'one-import-tf', 'one-import-tflite'
        ]
        # parse group option
        GROUP_OPTION_KEY = 'include'
        if self.cfgparser.has_option('onecc', GROUP_OPTION_KEY):
            groups = self.cfgparser['onecc'][GROUP_OPTION_KEY].split()
            for o in groups:
                if o == 'O' or not o.startswith('O'):
                    raise ValueError('Invalid group option')
                # add_opt receives group name except first 'O'
                self.add_opt(o[1:])

        self.backend = None
        self.target = None

    def _verify_cfg(self, cfgparser):
        if not cfgparser.has_section('onecc'):
            if cfgparser.has_section('one-build'):
                warnings.formatwarning = _simple_warning
                warnings.warn(
                    "[one-build] section will be deprecated. Please use [onecc] section.")
            else:
                raise ImportError('[onecc] section is required in configuration file')

    def _is_available(self, driver):
        # if there's no `onecc` section, it will find `one-build` section because of backward compatibility
        return (self.cfgparser.has_option('onecc', driver) and self.cfgparser.getboolean(
            'onecc', driver)) or (self.cfgparser.has_option('one-build', driver)
                                  and self.cfgparser.getboolean('one-build', driver))

    def add_opt(self, opt):
        self.optparser = oneutils.get_config_parser()
        opt_book = dict(
            zip(oneutils.get_optimization_list(get_name=True),
                oneutils.get_optimization_list()))
        parsed = self.optparser.read(opt_book['O' + opt])
        if not parsed:
            raise FileNotFoundError('Not found given optimization configuration file')
        if len(self.optparser.sections()) != 1 or self.optparser.sections(
        )[0] != 'one-optimize':
            raise AssertionError(
                'Optimization configuration file only allowed to have a \'one-optimize\' section'
            )
        self.opt = opt

    def set_backend(self, backend: str):
        self.backend = backend

    def set_target(self, target: str):
        self.target = target

    def detect_import_drivers(self, dir):
        self.import_drivers = list(oneutils.detect_one_import_drivers(dir).keys())

    def run(self, working_dir, verbose=False):
        # set environment
        CFG_ENV_SECTION = 'Environment'
        if self.cfgparser.has_section(CFG_ENV_SECTION):
            for key in self.cfgparser[CFG_ENV_SECTION]:
                os.environ[key] = self.cfgparser[CFG_ENV_SECTION][key]

        section_to_run = []
        for d in self.import_drivers + self.driver_sequence:
            if self._is_available(d):
                section_to_run.append(d)

        for section in section_to_run:
            options = ['--config', self.path, '--section', section]
            if section == 'one-optimize' and self.optparser:
                options += ['-O', self.opt]
            if verbose:
                options.append('--verbose')
            if (section == 'one-codegen' or section == 'one-profile') and self.backend:
                options += ['-b', self.backend]
            if self.target:
                options += ['-T', self.target]
            driver_path = os.path.join(working_dir, section)
            cmd = [driver_path] + options
            oneutils.run(cmd)
