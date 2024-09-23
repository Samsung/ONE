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
"""
This is for the command schema feature.

_one-cmds_ has lots of tools such as one-import, one-optimize, etc.
They have their own section in the configuration file and users can 
 give arguments with key-value pairs.
 
But, backend tools such as one-codegen and one-profile hasn't the same
 mechanism. Rather, they should pass all the arguments with `command` key
 because _onecc_ can't know the backends' interface in advance.

The command schema has been introduced for resolving these difficulties.
If users provide _onecc_ with the command schema that describes the interface 
 of the backend, users can give arguments with key-value paris like other tools.

NOTE. Command schema feature works only when target option is given.

[AS-IS]

# example.cfg
[backend]
target=my_target

[one-codegen]
backend=my_backend
commnad=--output sample.tvn sample.circle

[TO-BE]

# /usr/share/one/backends/command/my_backend/codegen.py
from onelib import argumentparse
from onelib.argumentparse import DriverName, NormalOption, TargetOption


def command_schema():
    parser = argumentparse.ArgumentParser()
    parser.add_argument("my_backend-compile", action=DriverName)
    parser.add_argument("--output", action=NormalOption)
    parser.add_argument("input", action=NormalOption)

    return parser

# /usr/share/one/target/my_target.ini
TARGET=my_target
BACKEND=my_backend

# example.cfg
[one-codegen]
output=sample.tvn
input=sample.circle


---

Command schema file should define `command_schema` function. And, you can add
 arguments by calling `add_argument`. You should specify an action according to
the option category.

[Action List]
- DriverName: the name of backend driver
- TargetOption: the target option of the drvier.
- NormalOption: the option of the driver. Starting with dash('-') implies the option
 is optional rather than positional one.
"""

import ntpath
from types import SimpleNamespace
from typing import List, Tuple, Union, Type
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


class Option():
    pass


class Positional(Option):
    pass


class Optional(Option):
    pass


class ArgumentParser():
    _SUPPORTED_ACTION_TYPE = [DriverName, NormalOption, TargetOption]

    def __init__(self):
        # List[args, action type, data type, option type]
        self._actions: List[Tuple[Tuple[str], Action, Union[Type[str],
                                                            Type[bool]]]] = list()
        self.driver: str = None
        self.target: str = None

    def print_help(self):
        backends_list = backends.get_list(self.driver)
        driver_path = None
        for cand in backends_list:
            if ntpath.basename(cand) == self.driver:
                driver_path = cand
        if not driver_path:
            driver_path = shutil.which(self.driver)

        if not driver_path:
            raise FileNotFoundError(self.driver + ' not found')

        oneutils.run([driver_path, '-h'], err_prefix=self.driver)

    def get_option_names(self, *, flatten=False, without_dash=False):
        """
        Get registered option names.

        :param flatten: single option can have multiple names. 
          If it is True, such options are returned after flattened.
        :param without_dash: optional argument has leading dash on its names. 
          If it is True, option names are returned without such dashes.

        For example, say there are options like these.

          parser.add_argument("--verbose", action=NormalOption, dtype=bool)
          parser.add_argument("--output", "--output_path", action=NormalOption)
        
        [EXAMPLES]
          get_option_names()
            [[--verbose], [--output, --output_path]]
          get_option_names(without_dash=True)
            [[verbose], [output, output_path]]
          get_option_names(flatten=True)
            [--verbose, --output, --output_path]
          get_option_names(flatten=True, without_dash=True)
            [verbose, output, output_path]
        """
        names = []
        for action in self._actions:
            names.append(action[0])

        if flatten:
            names = [name for name_l in names for name in name_l]
        if without_dash:
            names = [name.lstrip('-') for name in names]

        return names

    def check_if_valid_option_name(self, *args, **kwargs):
        existing_options = self.get_option_names(flatten=True, without_dash=True)
        args_without_dash = [arg.lstrip('-') for arg in args]
        if any(arg in existing_options for arg in args_without_dash):
            raise RuntimeError('Duplicate option names')
        if not 'action' in kwargs:
            raise RuntimeError('"action" keyword argument is required')

        action = kwargs['action']
        if not action in self._SUPPORTED_ACTION_TYPE:
            raise RuntimeError('Invalid action')
        if not args:
            raise RuntimeError('Invalid option name')
        if action == DriverName and len(args) >= 2:
            raise RuntimeError('onecc doesn\'t support multiple driver name')

        dtype = kwargs.get('dtype', str)
        if dtype == bool and action != NormalOption:
            raise RuntimeError('Only normal option can be boolean type')
        if dtype == bool and not all(a.startswith('-') for a in args):
            raise RuntimeError('Boolean type option should start with dash("-")')

    def add_argument(self, *args, **kwargs):
        self.check_if_valid_option_name(*args, **kwargs)

        action = kwargs['action']
        dtype = kwargs.get('dtype', str)
        if action == DriverName:
            assert len(args) == 1
            self.driver = args[0]
        else:
            if all(a.startswith('-') for a in args):
                otype = Optional
            elif all(not a.startswith('-') for a in args):
                otype = Positional
            else:
                raise RuntimeError(
                    'Invalid option names. Only either of option type is allowed: positional or optional'
                )
            self._actions.append((args, kwargs['action'], dtype, otype))

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
        invalid_options = list(cfg_args.__dict__.keys())
        # traverse the action in order and make commands
        for action in self._actions:
            args, act, dtype, otype = action
            assert act in [NormalOption, TargetOption]
            if otype == Optional:
                option_names = []
                for arg in args:
                    if arg.startswith('--'):
                        option_names.append(arg[len('--'):])
                    elif arg.startswith('-'):
                        option_names.append(arg[len('-'):])
            elif otype == Positional:
                option_names = args
            else:
                assert False

            given_option = None
            if act == NormalOption:
                for option_name in option_names:
                    if oneutils.is_valid_attr(cfg_args, option_name):
                        given_option = option_name
                        break
                if not given_option:
                    continue
                if dtype == bool and given_option.lower() == "false":
                    continue
            if otype == Optional:
                # use first option
                cmd += [args[0]]
            if act == TargetOption:
                cmd += [self.target]
            else:
                assert act == NormalOption
                if dtype == str:
                    cmd += [getattr(cfg_args, given_option)]
                invalid_options.remove(given_option)

        if len(invalid_options):
            print(f'WARNING: there are invalid options {invalid_options}')
            self.print_help()
        return cmd
