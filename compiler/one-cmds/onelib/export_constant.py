#!/usr/bin/env python

# Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

from constant import CONSTANT

import argparse
import configparser


def main():
    parser = argparse.ArgumentParser(
        description='Export CONSTANT value with given file format.')
    parser.add_argument('-c',
                        '--constant',
                        type=str,
                        required=True,
                        help='Constant name to export')
    parser.add_argument(
        '-f',
        '--format',
        type=str,
        required=True,
        choices=['cfg', 'txt'],
        help=
        'File format to export. The created cfg file contains CONSTANT under the one-optimize section.'
    )
    parser.add_argument(
        '--exclusive',
        action='store_true',
        help='Exports the rest of the options except for the given constant')
    parser.add_argument('-o',
                        '--output_path',
                        type=str,
                        required=True,
                        help='Path to output')

    args = parser.parse_args()

    if not hasattr(CONSTANT, args.constant):
        raise NameError('Not found given constant name')

    if args.exclusive:
        constant_to_exclude = getattr(CONSTANT, args.constant)
        constant_to_export = []
        for opt in CONSTANT.OPTIMIZATION_OPTS:
            if opt[0] in constant_to_exclude:
                continue
            constant_to_export.append(opt[0])
    else:
        constant_to_export = getattr(CONSTANT, args.constant)

    if args.format == 'cfg':
        SECTION_TO_EXPORT = 'one-optimize'
        config = configparser.ConfigParser()
        config[SECTION_TO_EXPORT] = dict()
        for constant in constant_to_export:
            config[SECTION_TO_EXPORT][constant] = 'True'

        with open(args.output_path, 'w') as f:
            config.write(f)

    if args.format == 'txt':
        with open(args.output_path, 'w') as f:
            for constant in constant_to_export:
                f.write(f"{constant}\n")


if __name__ == '__main__':
    main()
