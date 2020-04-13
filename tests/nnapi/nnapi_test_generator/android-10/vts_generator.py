#!/usr/bin/python3

# Copyright 2017, The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VTS testcase generator

Implements VTS test backend. Shares most logic with the CTS test
generator. Invoked by ml/nn/runtime/test/specs/generate_vts_tests.sh;
See that script for details on how this script is used.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from functools import reduce
import math
import numpy as np
import os
import re
import struct
import sys
import contextlib
import pprint

# Stuff from test generator
import test_generator as tg
from test_generator import ActivationConverter
from test_generator import BoolScalar
from test_generator import Configuration
from test_generator import DataTypeConverter
from test_generator import DataLayoutConverter
from test_generator import Example
from test_generator import Float16Scalar
from test_generator import Float32Scalar
from test_generator import Float32Vector
from test_generator import IgnoredOutput
from test_generator import Input
from test_generator import Int32Scalar
from test_generator import Int32Vector
from test_generator import Internal
from test_generator import Model
from test_generator import Operand
from test_generator import Output
from test_generator import Parameter
from test_generator import ParameterAsInputConverter
from test_generator import RelaxedModeConverter
from test_generator import SmartOpen
from test_generator import SymmPerChannelQuantParams

# Dumping methods that shared with CTS generator
from cts_generator import DumpCtsExample
from cts_generator import DumpCtsIsIgnored

# Take a model from command line
def ParseCmdLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="the spec file")
    parser.add_argument(
        "-m", "--model", help="the output model file", default="-")
    parser.add_argument(
        "-e", "--example", help="the output example file", default="-")
    parser.add_argument(
        "-t", "--test", help="the output test file", default="-")
    args = parser.parse_args()
    tg.FileNames.InitializeFileLists(
        args.spec, args.model, args.example, args.test)

# Generate operands in VTS format
def generate_vts_operands(model):
  # Dump operand definitions
  op_def = """\
        {{
            .type = OperandType::{operand_type},
            .dimensions = {shape},
            .numberOfConsumers = {no_consumers},
            .scale = {scale},
            .zeroPoint = {zero_point},
            .lifetime = OperandLifeTime::{lifetime},
            .location = {{.poolIndex = 0, .offset = {offset}, .length = {length}}},{extraParams}
        }}"""
  offset = 0
  op_definitions = []
  extra_params_definitions = []
  for index, o in enumerate(model.operands):
    length = o.type.GetByteSize() if isinstance(o, Parameter) else 0
    add_extra_params = o.type.extraParams is not None and not o.type.extraParams.hide
    op = {
        "operand_type": o.type.type,
        "shape": o.type.GetDimensionsString(),
        "no_consumers": len(o.outs),
        "scale": tg.PrettyPrintAsFloat(o.type.scale),
        "zero_point": str(int(o.type.zeroPoint)),
        "lifetime": o.lifetime,
        "offset": offset if isinstance(o, Parameter) else 0,
        "length": length,
        "extraParams": "" if not add_extra_params else "\n            .extraParams = std::move(extraParams%d)," % (index,),
    }
    offset += length
    op_definitions.append(op_def.format(**op))

    extra_params_def = """\
    Operand::ExtraParams extraParams{index};
    extraParams{index}.{setMethodName}({param});
"""

    if add_extra_params:
      ep = o.type.extraParams
      op = {
          "index": index,
          "setMethodName": ep.GetVtsSetter(),
          "param": ep.GetVtsConstructor(),
      }
      extra_params_definitions.append(extra_params_def.format(**op))

  op_vec = """{0}\
    const std::vector<Operand> operands = {{
{1}
    }};""".format(",\n".join(extra_params_definitions), ",\n".join(op_definitions))
  return op_vec

# Generate VTS operand values
def generate_vts_operand_values(operands):
    weights = [o for o in operands if isinstance(o, Parameter)]
    binit = []
    for w in weights:
        ty = w.type.type
        if ty == "TENSOR_QUANT8_ASYMM":
            binit += w.value
        elif ty == "TENSOR_QUANT8_SYMM_PER_CHANNEL" or ty == "TENSOR_QUANT8_SYMM":
            binit += [struct.pack("b", value)[0] for value in w.value]
        elif ty == "BOOL" or ty == "TENSOR_BOOL8":
            binit += [1 if x else 0 for x in w.value]
        elif ty == "TENSOR_FLOAT16" or ty == "FLOAT16":
            for f in w.value:
                # The pack format for float16 is not available until Python 3.6.
                binit += [int(x) for x in np.float16(f).tostring()]
        elif ty in {"TENSOR_FLOAT32", "FLOAT32", "TENSOR_INT32", "INT32", "TENSOR_QUANT16_ASYMM"}:
            if ty in ["TENSOR_FLOAT32", "FLOAT32"]:
                fmt = "f"
            elif ty in ["TENSOR_INT32", "INT32"]:
                fmt = "i"
            elif ty == "TENSOR_QUANT16_ASYMM":
                fmt = "H"
            for f in w.value:
                binit += [int(x) for x in struct.pack(fmt, f)]
        else:
            assert 0 and "Unsupported VTS operand type"

    init_defs = ", ".join([str(x) for x in binit])
    if (init_defs != ""):
        init_defs = "\n      %s\n    " % init_defs
    byte_vec_fmt = """{%s}""" % init_defs
    return byte_vec_fmt

# Generate VTS operations
def generate_vts_operation(op, model):
    op_fmt = """\
        {{
            .type = OperationType::{op_code},
            .inputs = {{{ins}}},
            .outputs = {{{outs}}},
        }}"""
    op_content = {
        'op_code': op.optype,
        'ins': tg.GetJointStr(model.GetIndexOfOperands(op.ins)),
        'outs': tg.GetJointStr(model.GetIndexOfOperands(op.outs))
    }
    return op_fmt.format(**op_content)

def generate_vts_operations(model):
    vts_ops = [generate_vts_operation(op, model) for op in model.operations]
    return ",\n".join(vts_ops)

def generate_vts_model(model, model_file):
  operand_values_fmt = ""
  if Configuration.useSHM():
    # Boilerplate code for passing weights in shared memory
    operand_values_fmt = """\
    std::vector<uint8_t> operandValues = {{}};
    const uint8_t data[] = {operand_values};

    // Allocate segment of android shared memory, wrapped in hidl_memory.
    // This object will be automatically freed when sharedMemory is destroyed.
    hidl_memory sharedMemory = allocateSharedMemory(sizeof(data));

    // Mmap ashmem into usable address and hold it within the mappedMemory object.
    // MappedMemory will automatically munmap the memory when it is destroyed.
    sp<IMemory> mappedMemory = mapMemory(sharedMemory);

    if (mappedMemory != nullptr) {{
        // Retrieve the mmapped pointer.
        uint8_t* mappedPointer =
                static_cast<uint8_t*>(static_cast<void*>(mappedMemory->getPointer()));

        if (mappedPointer != nullptr) {{
            // Acquire the write lock for the shared memory segment, upload the data,
            // and release the lock.
            mappedMemory->update();
            std::copy(data, data + sizeof(data), mappedPointer);
            mappedMemory->commit();
        }}
    }}

    const std::vector<hidl_memory> pools = {{sharedMemory}};
"""
  else:
    # Passing weights via operandValues
    operand_values_fmt = """\
    std::vector<uint8_t> operandValues = {operand_values};
    const std::vector<hidl_memory> pools = {{}};
"""

  operand_values_val = {
      'operand_values': generate_vts_operand_values(model.operands)
  }
  operand_values = operand_values_fmt.format(**operand_values_val)
  #  operand_values = operand_values_fmt
  model_fmt = """\
// Create the model
Model {create_test_model_name}() {{
{operand_decls}

    const std::vector<Operation> operations = {{
{operations}
    }};

    const std::vector<uint32_t> inputIndexes = {{{input_indices}}};
    const std::vector<uint32_t> outputIndexes = {{{output_indices}}};
{operand_values}
    return {{
        .operands = operands,
        .operations = operations,
        .inputIndexes = inputIndexes,
        .outputIndexes = outputIndexes,
        .operandValues = operandValues,
        .pools = pools,{relaxed_field}
    }};
}}
"""
  model_dict = {
      "create_test_model_name": str(model.createTestFunctionName),
      "operations": generate_vts_operations(model),
      "operand_decls": generate_vts_operands(model),
      "operand_values": operand_values,
      "output_indices": tg.GetJointStr(model.GetOutputsIndex()),
      "input_indices": tg.GetJointStr(model.GetInputsIndex()),
      "relaxed_field":
        "\n        .relaxComputationFloat32toFloat16 = true," if (model.isRelaxed) else ""
  }
  print(model_fmt.format(**model_dict), file = model_file)

def generate_vts(model, model_file):
  assert model.compiled
  generate_vts_model(model, model_file)
  DumpCtsIsIgnored(model, model_file)

def generate_vts_test(example, test_file):
    testTemplate = """\
TEST_F({test_case_name}, {test_name}) {{
  generated_tests::Execute(device,
                           {namespace}::{create_model_name},
                           {namespace}::{is_ignored_name},
                           {namespace}::get_{examples_name}(){test_dynamic_output_shape});\n}}

TEST_F(ValidationTest, {test_name}) {{
  const Model model = {namespace}::{create_model_name}();
  const std::vector<Request> requests = createRequests({namespace}::get_{examples_name}());
  validateEverything(model, requests);
}}\n
"""
    if example.model.hasDynamicOutputShape:
        print("#ifdef NN_TEST_DYNAMIC_OUTPUT_SHAPE", file=test_fd)
    print(testTemplate.format(
            test_case_name="DynamicOutputShapeTest" if example.model.hasDynamicOutputShape \
                           else "NeuralnetworksHidlTest",
            test_name=str(example.testName),
            namespace=tg.FileNames.specName,
            create_model_name=str(example.model.createTestFunctionName),
            is_ignored_name=str(example.model.isIgnoredFunctionName),
            examples_name=str(example.examplesName),
            test_dynamic_output_shape=", true" if example.model.hasDynamicOutputShape else ""
        ), file=test_fd)
    if example.model.hasDynamicOutputShape:
        print("#endif", file=test_fd)

def InitializeFiles(model_fd, example_fd, test_fd):
    fileHeader = "// clang-format off\n// Generated file (from: {spec_file}). Do not edit"
    testFileHeader = """\
// Generated from: {spec_file}.
namespace {spec_name} {{
// Generated {spec_name} test
#include "{example_file}"
// Generated model constructor
#include "{model_file}"
}} // namespace {spec_name}\n"""
    # This regex is to remove prefix and get relative path for #include
    pathRegex = r".*frameworks/ml/nn/(runtime/test/generated/)?"
    specFileBase = os.path.basename(tg.FileNames.specFile)
    print(fileHeader.format(spec_file=specFileBase), file=model_fd)
    print(fileHeader.format(spec_file=specFileBase), file=example_fd)
    print(testFileHeader.format(
        spec_file=specFileBase,
        model_file=re.sub(pathRegex, "", tg.FileNames.modelFile),
        example_file=re.sub(pathRegex, "", tg.FileNames.exampleFile),
        spec_name=tg.FileNames.specName), file=test_fd)

if __name__ == "__main__":
    ParseCmdLine()
    while tg.FileNames.NextFile():
        print("Generating test(s) from spec: %s" % tg.FileNames.specFile, file=sys.stderr)
        exec (open(tg.FileNames.specFile, "r").read())
        print("Output VTS model: %s" % tg.FileNames.modelFile, file=sys.stderr)
        print("Output example:" + tg.FileNames.exampleFile, file=sys.stderr)
        with SmartOpen(tg.FileNames.modelFile) as model_fd, \
             SmartOpen(tg.FileNames.exampleFile) as example_fd, \
             SmartOpen(tg.FileNames.testFile, mode="a") as test_fd:
            InitializeFiles(model_fd, example_fd, test_fd)
            Example.DumpAllExamples(
                DumpModel=generate_vts, model_fd=model_fd,
                DumpExample=DumpCtsExample, example_fd=example_fd,
                DumpTest=generate_vts_test, test_fd=test_fd)
