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

from onelib.constant import CONSTANT


class OptionBuilder:
    def __init__(self, one_cmd_type):
        self.type = one_cmd_type

    def _build_default(self, commands):
        options = []
        for k, v in commands.items():
            options.extend(['--' + k, v])
        return options

    def _build_with_unknown_command(self, commands):
        COMMAND_K = 'command'
        options = []
        for k, v in commands.items():
            if k == COMMAND_K:
                continue
            options.extend(['--' + k, v])
        options.extend(['--'])
        options.extend(commands[COMMAND_K].split())
        return options

    def _build_import(self, commands):
        options = []
        arg_0 = ['save_intermediate']
        for k, v in commands.items():
            if k in arg_0 and v == "True":
                options.extend(['--' + k])
                continue
            options.extend(['--' + k, v])
        return options

    def _build_optimize(self, commands):
        options = []
        arg_0 = ['generate_profile_data']
        arg_1 = ['input_path', 'output_path', 'change_outputs']
        for k, v in commands.items():
            if k in arg_1:
                options.extend(['--' + k, v])
                continue
            if k in arg_0 and v == 'True':
                options.extend(['--' + k])
                continue
            for opt in CONSTANT.OPTIMIZATION_OPTS:
                if k == opt[0] and v == "True":
                    options.extend(['--' + k])
                    break
        return options

    def _build_quantize(self, commands):
        options = []
        arg_0 = [
            'generate_profile_data', 'save_intermediate', 'TF-style_maxpool',
            'evaluate_result', 'print_mae', 'print_mape', 'print_mpeir',
            'print_top1_match', 'print_top5_match', 'force_quantparam', 'copy_quantparam'
        ]
        for k, v in commands.items():
            if k in arg_0 and v == "True":
                options.extend(['--' + k])
                continue
            options.extend(['--' + k, v])
        return options

    def build(self, commands):
        cmd_book = dict.fromkeys([
            'one-import-bcq', 'one-import-tflite', 'one-resize', 'one-pack',
            'one-partition'
        ], self._build_default)
        cmd_book['one-codegen'] = self._build_with_unknown_command
        cmd_book['one-import-onnx'] = self._build_import
        cmd_book['one-import-pytorch'] = self._build_import
        cmd_book['one-import-tf'] = self._build_import
        cmd_book['one-infer'] = self._build_with_unknown_command
        cmd_book['one-optimize'] = self._build_optimize
        cmd_book['one-profile'] = self._build_with_unknown_command
        cmd_book['one-quantize'] = self._build_quantize

        return cmd_book[self.type](commands)
