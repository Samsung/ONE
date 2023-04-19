/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci_interpreter/core/reader/CircleMicroReader.h"

#include <circle-generated/circle/schema_generated.h>

#include <iostream>
#include <fstream>
#include <set>

std::string get_register_kernel_str(const circle::BuiltinOperator builtin_operator)
{
  switch (builtin_operator)
  {
    case circle::BuiltinOperator_ADD:
      return "REGISTER_KERNEL(ADD, Add)";
    case circle::BuiltinOperator_ARG_MAX:
      return "REGISTER_KERNEL(ARG_MAX, ArgMax)";
    case circle::BuiltinOperator_AVERAGE_POOL_2D:
      return "REGISTER_KERNEL(AVERAGE_POOL_2D, AveragePool2D)";
    case circle::BuiltinOperator_BATCH_TO_SPACE_ND:
      return "REGISTER_KERNEL(BATCH_TO_SPACE_ND, BatchToSpaceND)";
    case circle::BuiltinOperator_CAST:
      return "REGISTER_KERNEL(CAST, Cast)";
    case circle::BuiltinOperator_CONCATENATION:
      return "REGISTER_KERNEL(CONCATENATION, Concatenation)";
    case circle::BuiltinOperator_CONV_2D:
      return "REGISTER_KERNEL(CONV_2D, Conv2D)";
    case circle::BuiltinOperator_DEPTH_TO_SPACE:
      return "REGISTER_KERNEL(DEPTH_TO_SPACE, DepthToSpace)";
    case circle::BuiltinOperator_DEPTHWISE_CONV_2D:
      return "REGISTER_KERNEL(DEPTHWISE_CONV_2D, DepthwiseConv2D)";
    case circle::BuiltinOperator_DEQUANTIZE:
      return "REGISTER_KERNEL(DEQUANTIZE, Dequantize)";
    case circle::BuiltinOperator_DIV:
      return "REGISTER_KERNEL(DIV, Div)";
    case circle::BuiltinOperator_ELU:
      return "REGISTER_KERNEL(ELU, Elu)";
    case circle::BuiltinOperator_EXP:
      return "REGISTER_KERNEL(EXP, Exp)";
    case circle::BuiltinOperator_EXPAND_DIMS:
      return "REGISTER_KERNEL(EXPAND_DIMS, ExpandDims)";
    case circle::BuiltinOperator_FILL:
      return "REGISTER_KERNEL(FILL, Fill)";
    case circle::BuiltinOperator_FLOOR:
      return "REGISTER_KERNEL(FLOOR, Floor)";
    case circle::BuiltinOperator_FLOOR_DIV:
      return "REGISTER_KERNEL(FLOOR_DIV, FloorDiv)";
    case circle::BuiltinOperator_EQUAL:
      return "REGISTER_KERNEL(EQUAL, Equal)";
    case circle::BuiltinOperator_FULLY_CONNECTED:
      return "REGISTER_KERNEL(FULLY_CONNECTED, FullyConnected)";
    case circle::BuiltinOperator_GREATER:
      return "REGISTER_KERNEL(GREATER, Greater)";
    case circle::BuiltinOperator_GREATER_EQUAL:
      return "REGISTER_KERNEL(GREATER_EQUAL, GreaterEqual)";
    case circle::BuiltinOperator_INSTANCE_NORM:
      return "REGISTER_KERNEL(INSTANCE_NORM, InstanceNorm)";
    case circle::BuiltinOperator_L2_NORMALIZATION:
      return "REGISTER_KERNEL(L2_NORMALIZATION, L2Normalize)";
    case circle::BuiltinOperator_L2_POOL_2D:
      return "REGISTER_KERNEL(L2_POOL_2D, L2Pool2D)";
    case circle::BuiltinOperator_LEAKY_RELU:
      return "REGISTER_KERNEL(LEAKY_RELU, LeakyRelu)";
    case circle::BuiltinOperator_LESS:
      return "REGISTER_KERNEL(LESS, Less)";
    case circle::BuiltinOperator_LESS_EQUAL:
      return "REGISTER_KERNEL(LESS_EQUAL, LessEqual)";
    case circle::BuiltinOperator_LOGICAL_AND:
      return "REGISTER_KERNEL(LOGICAL_AND, LogicalAnd)";
    case circle::BuiltinOperator_LOGICAL_NOT:
      return "REGISTER_KERNEL(LOGICAL_NOT, LogicalNot)";
    case circle::BuiltinOperator_LOGICAL_OR:
      return "REGISTER_KERNEL(LOGICAL_OR, LogicalOr)";
    case circle::BuiltinOperator_LOGISTIC:
      return "REGISTER_KERNEL(LOGISTIC, Logistic)";
    case circle::BuiltinOperator_GATHER:
      return "REGISTER_KERNEL(GATHER, Gather)";
    case circle::BuiltinOperator_MAXIMUM:
      return "REGISTER_KERNEL(MAXIMUM, Maximum)";
    case circle::BuiltinOperator_MAX_POOL_2D:
      return "REGISTER_KERNEL(MAX_POOL_2D, MaxPool2D)";
    case circle::BuiltinOperator_MINIMUM:
      return "REGISTER_KERNEL(MINIMUM, Minimum)";
    case circle::BuiltinOperator_MIRROR_PAD:
      return "REGISTER_KERNEL(MIRROR_PAD, MirrorPad)";
    case circle::BuiltinOperator_MUL:
      return "REGISTER_KERNEL(MUL, Mul)";
    case circle::BuiltinOperator_NEG:
      return "REGISTER_KERNEL(NEG, Neg)";
    case circle::BuiltinOperator_NOT_EQUAL:
      return "REGISTER_KERNEL(NOT_EQUAL, NotEqual)";
    case circle::BuiltinOperator_PAD:
      return "REGISTER_KERNEL(PAD, Pad)";
    case circle::BuiltinOperator_PADV2:
      return "REGISTER_KERNEL(PADV2, PadV2)";
    case circle::BuiltinOperator_PACK:
      return "REGISTER_KERNEL(PACK, Pack)";
    case circle::BuiltinOperator_PRELU:
      return "REGISTER_KERNEL(PRELU, PRelu)";
    case circle::BuiltinOperator_QUANTIZE:
      return "REGISTER_KERNEL(QUANTIZE, Quantize)";
    case circle::BuiltinOperator_REDUCE_PROD:
      return "REGISTER_KERNEL(REDUCE_PROD, ReduceCommon)";
    case circle::BuiltinOperator_RESHAPE:
      return "REGISTER_KERNEL(RESHAPE, Reshape)";
    case circle::BuiltinOperator_RESIZE_BILINEAR:
      return "REGISTER_KERNEL(RESIZE_BILINEAR, ResizeBilinear)";
    case circle::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
      return "REGISTER_KERNEL(RESIZE_NEAREST_NEIGHBOR, ResizeNearestNeighbor)";
    case circle::BuiltinOperator_RSQRT:
      return "REGISTER_KERNEL(RSQRT, Rsqrt)";
    case circle::BuiltinOperator_SHAPE:
      return "REGISTER_KERNEL(SHAPE, Shape)";
    case circle::BuiltinOperator_SOFTMAX:
      return "REGISTER_KERNEL(SOFTMAX, Softmax)";
    case circle::BuiltinOperator_SPACE_TO_BATCH_ND:
      return "REGISTER_KERNEL(SPACE_TO_BATCH_ND, SpaceToBatchND)";
    case circle::BuiltinOperator_SPACE_TO_DEPTH:
      return "REGISTER_KERNEL(SPACE_TO_DEPTH, SpaceToDepth)";
    case circle::BuiltinOperator_SLICE:
      return "REGISTER_KERNEL(SLICE, Slice)";
    case circle::BuiltinOperator_STRIDED_SLICE:
      return "REGISTER_KERNEL(STRIDED_SLICE, StridedSlice)";
    case circle::BuiltinOperator_SQRT:
      return "REGISTER_KERNEL(SQRT, Sqrt)";
    case circle::BuiltinOperator_SQUARE:
      return "REGISTER_KERNEL(SQUARE, Square)";
    case circle::BuiltinOperator_SQUARED_DIFFERENCE:
      return "REGISTER_KERNEL(SQUARED_DIFFERENCE, SquaredDifference)";
    case circle::BuiltinOperator_SQUEEZE:
      return "REGISTER_KERNEL(SQUEEZE, Squeeze)";
    case circle::BuiltinOperator_SUB:
      return "REGISTER_KERNEL(SUB, Sub)";
    case circle::BuiltinOperator_SVDF:
      return "REGISTER_KERNEL(SVDF, SVDF)";
    case circle::BuiltinOperator_SPLIT:
      return "REGISTER_KERNEL(SPLIT, Split)";
    case circle::BuiltinOperator_TANH:
      return "REGISTER_KERNEL(TANH, Tanh)";
    case circle::BuiltinOperator_TRANSPOSE:
      return "REGISTER_KERNEL(TRANSPOSE, Transpose)";
    case circle::BuiltinOperator_TRANSPOSE_CONV:
      return "REGISTER_KERNEL(TRANSPOSE_CONV, TransposeConv)";
    case circle::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      return "REGISTER_KERNEL(UNIDIRECTIONAL_SEQUENCE_LSTM, UnidirectionalSequenceLSTM)";
    case circle::BuiltinOperator_WHILE:
      return "REGISTER_KERNEL(WHILE, While)";
    default:
      assert(false && "Not supported kernel");
  }
}

std::vector<char> loadFile(const std::string &path)
{
  std::ifstream file(path, std::ios::binary | std::ios::in);
  if (!file.good())
  {
    assert(false && "Failed to open file");
  }

  file.unsetf(std::ios::skipws);

  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // reserve capacity
  std::vector<char> data(fileSize);

  // read the data
  file.read(data.data(), fileSize);
  if (file.fail())
  {
    assert(false && "Failed to read file");
  }

  return data;
}

// Parse model and write to std::ofstream &os models operations
void run(std::ofstream &os, const circle::Model *model)
{
  luci_interpreter::CircleReader reader;
  reader.parse(model);
  const uint32_t subgraph_size = reader.num_subgraph();

  // Set to avoid duplication in generated list
  std::set<circle::BuiltinOperator> operations_set;

  for (uint32_t g = 0; g < subgraph_size; g++)
  {
    reader.select_subgraph(g);
    auto ops = reader.operators();
    for (uint32_t i = 0; i < ops.size(); ++i)
    {
      const auto op = ops.at(i);
      auto op_builtin_operator = reader.builtin_code(op);

      auto result = operations_set.insert(op_builtin_operator);
      if (result.second)
      {
        os << get_register_kernel_str(op_builtin_operator) << std::endl;
      }
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    assert(false && "Should be 2 arguments: circle model path, and path for generated model\n");
  }

  std::string model_file(argv[1]);
  std::string generated_file_path(argv[2]);

  std::vector<char> model_data = loadFile(model_file);
  const circle::Model *circle_model = circle::GetModel(model_data.data());

  if (circle_model == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << model_file << "'" << std::endl;
    return 255;
  }

  // Open or create file
  std::ofstream out;
  out.open(generated_file_path);

  if (out.is_open())
    run(out, circle_model);
  else
    std::cout << "SMTH GOES WRONG WHILE OPEN FILE" << std::endl;
  return 0;
}
