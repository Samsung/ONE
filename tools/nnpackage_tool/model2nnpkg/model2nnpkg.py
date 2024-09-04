#!/usr/bin/env python3

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
import sys


def _is_json(myjson):
    try:
        json.load(myjson)
    except ValueError as e:
        return False
    return True


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
            raise Exception(f'error: {model_path} does not exist.')

        modelfile = os.path.basename(model_path)
        if len(modelfile.split('.')) == 1:
            raise Exception(
                'error: modelfile does not have extension.\n' +
                "Please provide extension so that $progname can identify what type of model you use."
            )

        if args.config:
            config_path = os.path.basename(args.config[i])
            if not os.path.isfile(config_path):
                raise Exception(f'error: {config_path} does not exist.')

    # Check each json file
    for io_info_path in [path for path in (args.io_info or [])]:
        with open(io_info_path, "r") as io_json:
            if not _is_json(io_json):
                raise Exception(
                    f'error: io info file {io_info_path} is not json file.\n' +
                    "Please provide json file that so that $progname can identify what inputs/outputs of model you use."
                )

    # Check size of indices of original model
    size_inputs = 0
    size_outputs = 0
    for model_index, io_info_path in enumerate([path for path in (args.io_info or [])]):
        with open(io_info_path, "r") as io_json:
            model_io = json.load(io_json)
            if model_index == 0:
                size_inputs = len(model_io["org-model-io"]["inputs"]["new-indices"])
                size_outputs = len(model_io["org-model-io"]["outputs"]["new-indices"])
            else:
                if size_inputs != len(model_io["org-model-io"]["inputs"]["new-indices"]):
                    raise Exception(
                        f'error: Invalid size of input indices\n' +
                        "The size of orginal model's inputs in io info file {io_info_path} is different from the previous files."
                    )
                if size_outputs != len(
                        model_io["org-model-io"]["outputs"]["new-indices"]):
                    raise Exception(
                        f'error: Invalid size of output indices.\n' +
                        "The size of orginal model's outputs in io info file {io_info_path} is different from the previous files."
                    )


def _get_args():
    parser = argparse.ArgumentParser(
        description='Convert model files (tflite, circle or tvn) to nnpkg.',
        usage=''' %(prog)s [options]
  Examples:
      %(prog)s -m add.tflite                           => create nnpkg "add" in current directory
      %(prog)s -o out -m add.tflite                    => create nnpkg "add" in out/
      %(prog)s -o out -p addpkg -m add.tflite          => create nnpkg "addpkg" in out/
      %(prog)s -c add.cfg -m add.tflite                => create nnpkg "add" with add.cfg
      %(prog)s -o out -p addpkg -m a1.tflite a2.tflite -i a1.json a2.json
        => create nnpkg "addpkg" with models a1.tflite and a2.tflite in out/
  ''')
    parser.add_argument('-o',
                        '--outdir',
                        type=str,
                        default=os.getcwd(),
                        metavar='output_directory',
                        help='set nnpkg output directory')
    parser.add_argument('-p',
                        '--nnpkg-name',
                        type=str,
                        metavar='nnpkg_name',
                        help='set nnpkg output name (default=[1st modelfile name])')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        nargs='+',
                        default='',
                        metavar='conf',
                        help='provide configuration files')
    parser.add_argument('-m',
                        '--models',
                        type=str,
                        nargs='+',
                        metavar='model',
                        help='provide model files')
    parser.add_argument('-i',
                        '--io-info',
                        type=str,
                        nargs='+',
                        metavar='io_info',
                        help='provide io info')

    args = parser.parse_args()

    _verify_args(args)

    if not args.nnpkg_name:
        first_model_name = os.path.basename(args.models[0]).rsplit('.', 1)[0]
        args.nnpkg_name = first_model_name

    args.prog = parser.prog

    return args


def _get_org_model_input_size(json_path):
    with open(json_path, "r") as io_json:
        model_io = json.load(io_json)
        return len(model_io["org-model-io"]["inputs"]["new-indices"])


def _get_org_model_output_size(json_path):
    with open(json_path, "r") as io_json:
        model_io = json.load(io_json)
        return len(model_io["org-model-io"]["outputs"]["new-indices"])


def _generate_io_conn_info(io_info_files):
    ret = {}

    if io_info_files is None:
        return ret

    pkg_inputs = list(range(_get_org_model_input_size(io_info_files[0])))
    pkg_outputs = list(range(_get_org_model_output_size(io_info_files[0])))

    org_model_io = []
    new_model_io = {"inputs": [], "outputs": []}
    for model_pos, io_info_path in enumerate(io_info_files):
        with open(io_info_path, "r") as io_json:
            model_io = json.load(io_json)

            org_model_io.append(model_io["org-model-io"])
            new_model_io["inputs"].append(model_io["new-model-io"]["inputs"])
            new_model_io["outputs"].append(model_io["new-model-io"]["outputs"])

    for model_pos in range(len(org_model_io)):
        # Set pkg-inputs
        for org_model_input_pos, new_input_index in enumerate(
                org_model_io[model_pos]["inputs"]["new-indices"]):
            if new_input_index != -1:
                for new_model_input_pos, input_index in enumerate(
                        new_model_io["inputs"][model_pos]["new-indices"]):
                    if new_input_index == input_index:
                        pkg_inputs[
                            org_model_input_pos] = f'{model_pos}:0:{new_model_input_pos}'
                        break

                if pkg_inputs[org_model_input_pos] == 0:
                    raise Exception(
                        f'error: Wrong io information\n' +
                        "The input index {new_input_index} exists in org-model-io, but not in new-model-io\n"
                        + "Please check {io_info_files[model_pos]}")

        # Set pkg-outputs
        for org_model_output_pos, new_output_index in enumerate(
                org_model_io[model_pos]["outputs"]["new-indices"]):
            if new_output_index != -1:
                for new_model_output_pos, output_index in enumerate(
                        new_model_io["outputs"][model_pos]["new-indices"]):
                    if new_output_index == output_index:
                        pkg_outputs[
                            org_model_output_pos] = f'{model_pos}:0:{new_model_output_pos}'
                        break

                if pkg_outputs[org_model_output_pos] == 0:
                    raise Exception(
                        f'error: Wrong io information\n' +
                        "The output index {new_output_index} exists in org-model-io, but not in new-model-io\n"
                        + "Please check {io_info_files[model_pos]}")

    ret["pkg-inputs"] = pkg_inputs
    ret["pkg-outputs"] = pkg_outputs

    model_connect = {}
    for input_model_pos, inputs in enumerate(new_model_io["inputs"]):
        for output_model_pos, outputs in enumerate(new_model_io["outputs"]):
            if input_model_pos == output_model_pos:
                continue

            for input_index_pos, org_input_index in enumerate(inputs["org-indices"]):
                for output_index_pos, org_output_index in enumerate(
                        outputs["org-indices"]):
                    if org_input_index == org_output_index:
                        edge_to = f'{input_model_pos}:0:{input_index_pos}'
                        edge_from = f'{output_model_pos}:0:{output_index_pos}'

                        if edge_from not in model_connect:
                            model_connect[edge_from] = [edge_to]
                        else:
                            model_connect[edge_from].append(edge_to)

    ret["model-connect"] = [{
        "from": edge_from,
        "to": edge_to
    } for edge_from, edge_to in model_connect.items()]

    return ret


def _generate_manifest(args):
    config_list = [""]
    if args.config:
        config_list = [os.path.basename(e) for e in args.config]
    models_list = [os.path.basename(e) for e in args.models]
    types_list = [os.path.basename(e).rsplit('.', 1)[1] for e in args.models]
    io_conn_info = _generate_io_conn_info(args.io_info)

    manifest = {}
    manifest["major-version"] = "1"
    manifest["minor-version"] = "2"
    manifest["patch-version"] = "0"
    manifest["configs"] = config_list
    manifest["models"] = models_list
    manifest["model-types"] = types_list
    manifest = {**manifest, **io_conn_info}  # Requires python 3.5 or greater

    return manifest


def main():
    try:
        # parse arguments
        args = _get_args()

        print(f'{args.prog}: Generating nnpkg {args.nnpkg_name} in {args.outdir}')
        # mkdir nnpkg directory
        nnpkg_path = os.path.join(args.outdir, args.nnpkg_name)
        os.makedirs(os.path.join(nnpkg_path, 'metadata'), exist_ok=True)

        # dump manifest file
        manifest = _generate_manifest(args)
        manifest_path = os.path.join(nnpkg_path, 'metadata', 'MANIFEST')
        with open(manifest_path, "w") as json_file:
            json_file.write(f'{json.dumps(manifest, indent=2)}\n')

        # copy models and configurations
        for i in range(len(args.models)):
            shutil.copy2(args.models[i], nnpkg_path)
            if args.config:
                shutil.copy2(args.config[i], os.path.join(nnpkg_path, 'metadata'))
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
