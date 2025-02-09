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


def splitsubgraph_ios(iofile):
    iolist = re.split('--input-name |;--output-name ', iofile)
    in_ = iolist[1].split(';')
    out_ = iolist[2].split(';')
    del out_[-1]
    type = iolist[0].split('subgraph')[0]
    return in_, out_, type


def split_onnx_ios(instrfile, input_path='./resnet-test.onnx', out_folder='subgraphs/'):
    if not os.path.exists(input_path):
        print(input_path + " not exist")
        return

    model = onnx.load(input_path)
    onnx.checker.check_model(input_path)
    for output in model.graph.output:
        model.graph.value_info.append(output)
    onnx.save(model, input_path)
    f1 = open(instrfile, "r")
    lines = f1.readlines()
    cpu_count = 0
    npu_count = 0
    count = 0
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for line in lines:
        input_names, output_names, type = splitsubgraph_ios(line)
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
    f1.close()


if __name__ == "__main__":
    split_onnx_ios('./scripts/subgraphs_ios.txt', './resnet-test.onnx')
