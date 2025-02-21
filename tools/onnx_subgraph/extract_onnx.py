# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import onnx
import re
import os
import argparse


def split_subgraph_ios(iofile):
    iolist = re.split('--input-name |;--output-name ', iofile)
    in_ = iolist[1].split(';')
    out_ = iolist[2].split(';')
    del out_[-1]
    type = iolist[0].split('subgraph')[0]
    return in_, out_, type


def split_onnx_ios(instrfile, input_path, out_folder='subgraphs/'):
    os.makedirs(out_folder, exist_ok=True)

    model = onnx.load(input_path)
    onnx.checker.check_model(input_path)
    for output in model.graph.output:
        model.graph.value_info.append(output)
    onnx.save(model, input_path)

    try:
        with open(instrfile, "r") as f1:
            lines = f1.readlines()
    except Exception as e:
        print(e)
        raise

    cpu_count = 0
    npu_count = 0
    count = 0

    for line in lines:
        input_names, output_names, type = split_subgraph_ios(line)
        if (type == 'CPU'):
            count = cpu_count
            cpu_count = cpu_count + 1
        else:
            count = npu_count
            npu_count = npu_count + 1

        output_path_folder = out_folder
        if not os.path.exists(output_path_folder):
            os.makedirs(output_path_folder)
        output_path = output_path_folder + type + 'subgraph' + str(count) + '.onnx'

        if ((input_names != ['']) and (output_names != [''])):
            onnx.utils.extract_model(input_path, output_path, input_names, output_names)
            print("succeed", count)
            count = count + 1


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s',
                            '--subio',
                            default='./scripts/subgraphs_ios.txt',
                            help="set subgraphs input/output node information")
    arg_parser.add_argument('-m',
                            '--model',
                            default='./resnet-test.onnx',
                            help="set onnx model path")
    args = arg_parser.parse_args()

    split_onnx_ios(args.subio, args.model)
