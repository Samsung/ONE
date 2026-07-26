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

import torch
import onnx
import re
import os


def splitinstruction(instr):
    iolist = re.split('--input-name \"|\" --output-name \"|\" --input-shape \"', instr)
    del iolist[0]
    del iolist[-1]
    in_ = iolist[0].split(';')
    out_ = iolist[1].split(';')
    return in_, out_


def splitsubgraph_ios(iofile):
    iolist = re.split('--input-name |;--output-name ', iofile)
    in_ = iolist[1].split(';')
    out_ = iolist[2].split(';')
    del out_[-1]
    type = iolist[0].split('subgraph')[0]
    return in_, out_, type


def split_onnx(instrfile, type):
    print("module found")
    f1 = open(instrfile, "r")
    lines = f1.readlines()
    count = 0
    for line in lines:
        input_names, output_names = splitinstruction(line)
        input_path = 'net/diffusion_model_fp32_with_shape.onnx'
        output_path = 'diffusion_model_fp32_subgraphs_' + type + '/' + type + 'subgraph' + str(
            count) + '.onnx'
        count = count + 1
        if ((input_names != ['']) and (output_names != [''])):
            onnx.utils.extract_model(input_path, output_path, input_names, output_names)
    f1.close()


def split_onnx_ios(instrfile,
                   input_path='net/generation_model_simplify.onnx',
                   out_folder='subgraphs/'):
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


def rename_node_io(file_path):
    model = onnx.load(file_path)
    graph = model.graph
    for inputs in graph.input:
        inputs.name = re.sub(r'[/.]', '', inputs.name)
    for outputs in graph.output:
        outputs.name = re.sub(r'[/.]', '', outputs.name)
    for value_infos in graph.value_info:
        value_infos.name = re.sub(r'[/.]', '', value_infos.name)
    for initializers in graph.initializer:
        initializers.name = re.sub(r'[/.]', '', initializers.name)
    for node in graph.node:
        node.name = re.sub(r'[/.]', '', node.name)
        for i in range(len(node.input)):
            node.input[i] = re.sub(r'[/.]', '', node.input[i])
        for i in range(len(node.output)):
            node.output[i] = re.sub(r'[/.]', '', node.output[i])
    return model


def rename_subgraph_node_ios(in_file_path, out_file_path):
    file_names = os.listdir(in_file_path)
    for filename in file_names:
        filename_ = in_file_path + '/' + filename
        model = rename_node_io(filename_)
        output_file_path = out_file_path + '/' + filename
        onnx.save(model, output_file_path)
        print(f'Modified model saved to {output_file_path}')


def print_model(file_path):
    model = onnx.load(file_path)
    graph = model.graph
    size = 0
    for node in graph.node:
        size = size + 1
    print(size)


def sort(ifile_path, ofile_path):
    finished_flag = 0
    sort_count = 0
    f1 = open(ifile_path, "r")
    lines = f1.readlines()
    graphs_inputs = {}
    graphs_outputs = {}
    order_Subgraphs = {}
    issort_Subgraphs = {}
    TYPE = {}
    index = 0
    for line in lines:
        input_names, output_names, type = splitsubgraph_ios(line)
        graphs_inputs[index] = input_names
        graphs_outputs[index] = output_names
        TYPE[index] = type
        index = index + 1
    graph_num = index
    f1.close()
    while finished_flag == 0:
        finished_flag = 1
        if (sort_count) == 0:
            for i in range(graph_num):
                find_flag = 0
                for g_input in graphs_inputs[i]:
                    for j in range(graph_num):
                        if g_input in graphs_outputs[j]:
                            find_flag = 1
                            break
                    if find_flag == 1:
                        break
                if find_flag == 0:
                    order_Subgraphs[i] = 0
                    issort_Subgraphs[i] = 1
                else:
                    order_Subgraphs[i] = 1
                    issort_Subgraphs[i] = 0
                    finished_flag = 0
        else:
            for i in range(graph_num):
                find_flag = 0
                if issort_Subgraphs[i] == 1:
                    continue
                for g_input in graphs_inputs[i]:
                    for j in range(graph_num):
                        if g_input in graphs_outputs[j]:
                            if issort_Subgraphs[j] == 0:
                                find_flag = 1
                            break
                    if find_flag == 1:
                        break
                if find_flag == 0:
                    order_Subgraphs[i] = sort_count
                    issort_Subgraphs[i] = 1
                else:
                    order_Subgraphs[i] = sort_count + 1
                    issort_Subgraphs[i] = 0
                    finished_flag = 0
                if i == graph_num - 1:
                    for j in range(graph_num):
                        if order_Subgraphs[j] == sort_count:
                            issort_Subgraphs[j] = 1
        print(order_Subgraphs)
        print(issort_Subgraphs)
        sort_count = sort_count + 1
        f2 = open(ofile_path, "w")
        count_cpu = 0
        count_npu = 0
        for i in range(graph_num):
            content = ""
            if TYPE[i] == 'CPU':
                content = "CPUsubgraph" + str(count_cpu) + ": order" + str(
                    order_Subgraphs[i]) + "--input-name "
                count_cpu = count_cpu + 1
            if TYPE[i] == 'NPU':
                content = "NPUsubgraph" + str(count_npu) + ": order" + str(
                    order_Subgraphs[i]) + "--input-name "
                count_npu = count_npu + 1
            for graph_input in graphs_inputs[i]:
                content = content + graph_input + ";"
            content = content + "--output-name "
            for graph_output in graphs_outputs[i]:
                content = content + graph_output + ";"
            content = content + "\n"
            print(content)
            f2.write(content)
        f2.close()
