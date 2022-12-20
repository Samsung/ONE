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

import argparse
import json
import os
import shutil


def _verify_args(args):
    if args.config and len(args.config) != len(args.models):
        raise Exception(
            'error: when config file is provided, # of config file should be same with modelfile\n'
            +
            "Please provide config file for each model file, or don't provide config file."
        )

    for i in range(len(args.models)):
        model_path = args.models[i]
        if not os.path.isfile(model_path):
            raise Exception('error: modelfile does not have extension.')

        modelfile = os.path.basename(model_path)
        if len(modelfile.split('.')) == 1:
            raise Exception(
                'error: modelfile does not have extension.\n' +
                "Please provide extension so that $progname can identify what type of model you use."
            )

        if args.config:
            config_path = os.path.basename(args.config[i])
            if not os.path.isfile(config_path):
                raise Exception('error: {} does not exist.'.format(config_path))


def _get_args():
    parser = argparse.ArgumentParser(
        description='Convert model files (tflite, circle or tvn) to nnpackage.',
        usage=''' %(prog)s [options]
  Examples:
      %(prog)s -m add.tflite                           => create nnpackage "add" in current directory
      %(prog)s -o out -m add.tflite                    => create nnpackage "add" in out/
      %(prog)s -o out -p addpkg -m add.tflite          => create nnpackage "addpkg" in out/
      %(prog)s -c add.cfg -m add.tflite                => create nnpackage "add" with add.cfg
      %(prog)s -o out -p addpkg -m a1.tflite a2.tflite => create nnpackage "addpkg" with models a1.tflite and a2.tflite in out/
  ''')
    parser.add_argument(
        '-o',
        '--outdir',
        type=str,
        default=os.getcwd(),
        metavar='output_directory',
        help='set nnpackage output directory')
    parser.add_argument(
        '-p',
        '--nnpackage-name',
        type=str,
        metavar='nnpackage_name',
        help='set nnpackage output name (default=[1st modelfile name])')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        nargs='+',
        default='',
        metavar='conf',
        help='provide configuration files')
    parser.add_argument(
        '-m',
        '--models',
        type=str,
        nargs='+',
        metavar='model',
        help='provide model files')

    args = parser.parse_args()

    _verify_args(args)

    if not args.nnpackage_name:
        first_model_name = os.path.basename(args.models[0]).rsplit('.', 1)[0]
        args.nnpackage_name = first_model_name

    args.prog = parser.prog

    return args


def _generate_manifest(args):
    config_list = ''
    if args.config:
        config_list = [os.path.basename(e) for e in args.config]
    models_list = [os.path.basename(e) for e in args.models]
    types_list = [os.path.basename(e).rsplit('.', 1)[1] for e in args.models]
    # types_str = ','.join("{}".format(os.path.basename(e).rsplit('.', 1)[1]) for e in args.models)

    manifest = {}
    manifest["major-version"] = "1"
    manifest["minor-version"] = "2"
    manifest["patch-version"] = "0"
    manifest["configs"] = config_list
    manifest["models"] = models_list
    manifest["model-types"] = types_list

    return manifest


def main():
    # parse arguments
    args = _get_args()
    print(args)

    print('{}: Generating nnpackage {} in {}'.format(args.prog, args.nnpackage_name,
                                                     args.outdir))
    # mkdir nnpackage directory
    nnpackage_path = os.path.join(args.outdir, args.nnpackage_name)
    os.makedirs(os.path.join(nnpackage_path, 'metadata'), exist_ok=True)

    # dump menifest file
    manifest = _generate_manifest(args)
    menifest_path = os.path.join(nnpackage_path, 'metadata', 'MANIFEST')
    with open(menifest_path, "w") as json_file:
        json.dump(manifest, json_file, sort_keys=True, indent=2)

    # copy models and configures
    for i in range(len(args.models)):
        shutil.copy2(args.models[i], nnpackage_path)
        if args.config:
            shutil.copy2(args.config[i], os.path.join(nnpackage_path, 'metadata'))


if __name__ == "__main__":
    main()
