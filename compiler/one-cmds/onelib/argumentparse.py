#!/usr/bin/env python

# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

from types import SimpleNamespace
from typing import List, Tuple
import shutil

import onelib.backends as backends
import onelib.utils as oneutils


class Action():
    pass


class DriverName(Action):
    pass


class NormalOption(Action):
    pass


class TargetOption(Action):
    pass


class ArgumentParser():
    _SUPPORTED_ACTIONS = [DriverName, NormalOption, TargetOption]

    def __init__(self):
        self._actions: List[Tuple[str, Action]] = list()
        self.driver: str = None
        self.target: str = None

    def add_argument(self, *args, **kwargs):
        if not 'action' in kwargs:
            raise RuntimeError('"action" keyword argument is required')
        action = kwargs['action']
        if not action in self._SUPPORTED_ACTIONS:
            raise RuntimeError('Invalid action')
        if not args:
            raise RuntimeError('Invalid option name')

        # use first option.
        arg = args[0]
        if action == DriverName:
            self.driver = arg
        else:
            self._actions.append((arg, kwargs['action']))

    def make_cmd(self, cfg_args: SimpleNamespace) -> List:
        assert self.target, "Target should be set before making commands"
        assert self.driver, "Driver should be set before making commands"
        # find driver path
        driver_name = self.driver
        driver_list = backends.get_list(driver_name)
        if not driver_list:
            driver_list = [shutil.which(driver_name)]
            if not driver_list:
                raise FileNotFoundError(f'{driver_name} not found')
        # use first driver
        driver_path = driver_list[0]
        cmd: List = [driver_path]
        # traverse the action in order and make commands
        for action in self._actions:
            arg, act = action
            assert act in [NormalOption, TargetOption]
            # positional input doesn't have dash(-) in the string
            option_name = arg
            # optional input
            if arg.startswith('--'):
                option_name = arg[len('--'):]
            elif arg.startswith('-'):
                option_name = arg[len('-'):]

            if act == NormalOption and not oneutils.is_valid_attr(cfg_args, option_name):
                # TODO raise error when invalid option is given in the cfg file.
                continue
            if arg.startswith('-'):
                cmd += [arg]
            if act == TargetOption:
                cmd += [self.target]
            else:
                assert act == NormalOption
                cmd += [getattr(cfg_args, option_name)]
        return cmd
