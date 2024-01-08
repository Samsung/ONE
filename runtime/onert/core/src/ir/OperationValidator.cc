/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OperationValidator.h"

#include "ir/Graph.h"
#include "util/logging.h"

#define OP_REQUIRES(EXP)                                                                         \
  do                                                                                             \
  {                                                                                              \
    if (!(EXP))                                                                                  \
      throw std::runtime_error("OperationValidator failed at line " + std::to_string(__LINE__)); \
  } while (0)

namespace onert
{
namespace ir
{

OperationValidator::OperationValidator(const Graph &graph)
  : _operations{graph.operations()}, _operands{graph.operands()}
{
}

void OperationValidator::operator()()
{
  _operations.iterate([&](const OperationIndex &, const IOperation &node) { node.accept(*this); });
}

DataType OperationValidator::operandType(const OperandIndex &idx)
{
  return _operands.at(idx).typeInfo().type();
}

bool OperationValidator::isConstant(const OperandIndex &idx)
{
  return _operands.at(idx).isConstant();
}

bool OperationValidator::isSameType(const OperandIndex &idx1, const OperandIndex &idx2)
{
  return operandType(idx1) == operandType(idx2);
}

bool OperationValidator::isSameQuantParam(const OperandIndex &idx1, const OperandIndex &idx2)
{
  if (_operands.at(idx1).typeInfo().scale() != _operands.at(idx2).typeInfo().scale())
    return false;

  if (_operands.at(idx1).typeInfo().zero_point() != _operands.at(idx2).typeInfo().zero_point())
    return false;

  return true;
}

bool OperationValidator::isValidType(const OperandIndex &idx, const DataType &type)
{
  return operandType(idx) == type;
}

bool OperationValidator::isValidType(const OperandIndex &idx,
                                     std::initializer_list<DataType> valid_types)
{
  for (auto &&type_to_check : valid_types)
  {
    if (isValidType(idx, type_to_check))
    {
      return true;
    }
  }

  return false;
}

void OperationValidator::visit(const operation::AddN &node)
{
  const auto output_index(node.getOutputs().at(0));

  int size = node.getInputs().size();
  for (int i = 0; i < size; i++)
  {
    const auto input_index(node.getInputs().at(i));
    OP_REQUIRES(isValidType(input_index, {DataType::FLOAT32, DataType::INT32}));
    OP_REQUIRES(isSameType(input_index, output_index));
  }
}

void OperationValidator::visit(const operation::ArgMinMax &node)
{
  const auto input_index(node.getInputs().at(operation::ArgMinMax::Input::INPUT));
  const auto axis_index(node.getInputs().at(operation::ArgMinMax::Input::AXIS));
  const auto output_index(node.getOutputs().at(0));
  const auto output_type = node.param().output_type;

  OP_REQUIRES(isValidType(input_index, {DataType::FLOAT32, DataType::INT32, DataType::UINT8,
                                        DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_ASYMM}));
  OP_REQUIRES(isValidType(axis_index, {DataType::INT32, DataType::INT64}));
  OP_REQUIRES(isValidType(output_index, {DataType::INT32, DataType::INT64}));
  OP_REQUIRES(isValidType(output_index, output_type));
}

void OperationValidator::visit(const operation::BatchMatMul &node)
{
  const auto lhs_index(node.getInputs().at(operation::BatchMatMul::Input::LHS));
  const auto rhs_index(node.getInputs().at(operation::BatchMatMul::Input::RHS));
  const auto output_index(node.getOutputs().at(0));

  // Constant lhs and rhs is not implemented yet
  OP_REQUIRES(!isConstant(lhs_index) && !isConstant(rhs_index));

  // Allow hybrid quantization (lhs: float / rhs: qint8 / out: float)
  OP_REQUIRES(isValidType(
    lhs_index, {DataType::FLOAT32, DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_ASYMM}));
  OP_REQUIRES(isSameType(lhs_index, rhs_index) ||
              ((operandType(lhs_index) == DataType::FLOAT32) &&
               (operandType(rhs_index) == DataType::QUANT_INT8_ASYMM)));
  OP_REQUIRES(isSameType(lhs_index, output_index));
}

void OperationValidator::visit(const operation::BatchToSpaceND &node)
{
  const auto input_index{node.getInputs().at(operation::BatchToSpaceND::Input::INPUT)};
  const auto output_index{node.getOutputs().at(0)};

  OP_REQUIRES(isSameType(input_index, output_index));
}

void OperationValidator::visit(const operation::BinaryArithmetic &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(operation::BinaryArithmetic::Input::RHS)};

  OP_REQUIRES(isSameType(lhs_index, rhs_index));
  OP_REQUIRES(isSameType(lhs_index, output_index));
}

void OperationValidator::visit(const operation::Comparison &node)
{
  const auto output_index{node.getOutputs().at(0)};

  const auto lhs_index{node.getInputs().at(operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(operation::Comparison::Input::INPUT1)};

  OP_REQUIRES(isSameType(lhs_index, rhs_index));
  OP_REQUIRES(isValidType(output_index, DataType::BOOL8));
}

void OperationValidator::visit(const operation::Concat &node)
{
  const auto output_index{node.getOutputs().at(0)};

  for (auto &&input_index : node.getInputs())
  {
    OP_REQUIRES(isSameType(input_index, output_index));

    // Int8 quantization requires same scale and zero point
    if (isValidType(output_index, DataType::QUANT_INT8_ASYMM))
    {
      OP_REQUIRES(isSameQuantParam(input_index, output_index));
    }
  }
}

void OperationValidator::visit(const operation::Conv2D &node)
{
  const auto input_index{node.getInputs().at(operation::Conv2D::Input::INPUT)};
  const auto kernel_index{node.getInputs().at(operation::Conv2D::Input::KERNEL)};
  const auto output_index{node.getOutputs().at(0)};

  uint32_t stride_horizontal = node.param().stride.horizontal;
  uint32_t stride_vertical = node.param().stride.vertical;
  uint32_t dilation_width = node.param().dilation.width_factor;
  uint32_t dilation_height = node.param().dilation.height_factor;

  OP_REQUIRES((stride_horizontal > 0) && (stride_vertical > 0));
  OP_REQUIRES((dilation_width > 0) && (dilation_height > 0));
  OP_REQUIRES(isSameType(input_index, output_index));

  if (isConstant(kernel_index) && operandType(kernel_index) == DataType::QUANT_INT8_ASYMM)
  {
    for (const auto zeropoint : _operands.at(kernel_index).typeInfo().zero_points())
      OP_REQUIRES(zeropoint == 0);
  }
}

void OperationValidator::visit(const operation::DepthToSpace &node)
{
  const auto input_index{node.getInputs().at(operation::DepthToSpace::Input::INPUT)};
  const auto output_index{node.getOutputs().at(0)};

  int32_t block_size = node.param().block_size;

  OP_REQUIRES(isValidType(input_index, {DataType::FLOAT32, DataType::INT32, DataType::INT64,
                                        DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_ASYMM}));
  OP_REQUIRES(isSameType(input_index, output_index));

  OP_REQUIRES(block_size > 0);
}

void OperationValidator::visit(const operation::DetectionPostProcess &node)
{
  const auto &param = node.param();

  // FIXME: number of classes should be 1 for now.
  OP_REQUIRES(param.num_classes == 1);
}

void OperationValidator::visit(const operation::DepthwiseConv2D &node)
{
  const auto input_index{node.getInputs().at(operation::DepthwiseConv2D::Input::INPUT)};
  const auto kernel_index{node.getInputs().at(operation::DepthwiseConv2D::Input::KERNEL)};
  const auto output_index{node.getOutputs().at(0)};

  uint32_t stride_horizontal = node.param().stride.horizontal;
  uint32_t stride_vertical = node.param().stride.vertical;
  uint32_t dilation_width = node.param().dilation.width_factor;
  uint32_t dilation_height = node.param().dilation.height_factor;

  OP_REQUIRES((stride_horizontal > 0) && (stride_vertical > 0));
  OP_REQUIRES((dilation_width > 0) && (dilation_height > 0));
  OP_REQUIRES(isSameType(input_index, output_index));

  if (isConstant(kernel_index) && operandType(kernel_index) == DataType::QUANT_INT8_ASYMM)
  {
    for (const auto zeropoint : _operands.at(kernel_index).typeInfo().zero_points())
      OP_REQUIRES(zeropoint == 0);
  }
}

void OperationValidator::visit(const operation::ElementwiseActivation &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  // Check if I/O types match
  OP_REQUIRES(isSameType(output_index, input_index));

  switch (node.param().op_type)
  {
    case operation::ElementwiseActivation::Type::ELU:
      OP_REQUIRES(isValidType(input_index, DataType::FLOAT32));
      break;
    case operation::ElementwiseActivation::Type::LEAKY_RELU:
      OP_REQUIRES(
        isValidType(input_index, {DataType::FLOAT32, DataType::QUANT_UINT8_ASYMM,
                                  DataType::QUANT_INT8_ASYMM, DataType::QUANT_INT16_ASYMM}));
      break;
    case operation::ElementwiseActivation::Type::LOGISTIC:
      OP_REQUIRES(
        isValidType(input_index, {DataType::FLOAT32, DataType::QUANT_UINT8_ASYMM,
                                  DataType::QUANT_INT8_ASYMM, DataType::QUANT_INT16_ASYMM}));
      break;
    case operation::ElementwiseActivation::Type::RELU:
      OP_REQUIRES(isValidType(
        input_index, {DataType::FLOAT32, DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_ASYMM}));
      break;
    case operation::ElementwiseActivation::Type::TANH:
      OP_REQUIRES(
        isValidType(input_index, {DataType::FLOAT32, DataType::QUANT_UINT8_ASYMM,
                                  DataType::QUANT_INT8_ASYMM, DataType::QUANT_INT16_ASYMM}));
      break;
  }
}

void OperationValidator::visit(const operation::ElementwiseBinary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(operation::ElementwiseBinary::Input::LHS)};
  const auto rhs_index{node.getInputs().at(operation::ElementwiseBinary::Input::RHS)};

  OP_REQUIRES(isSameType(lhs_index, rhs_index));
  OP_REQUIRES(isSameType(lhs_index, output_index));

  const auto op_type = node.param().op_type;
  if (op_type == operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_AND ||
      op_type == operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_OR)
  {
    OP_REQUIRES(isValidType(lhs_index, DataType::BOOL8));
  }
}

void OperationValidator::visit(const operation::ElementwiseUnary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(operation::ElementwiseUnary::Input::INPUT)};

  // Check if I/O types match
  if (node.param().op_type == operation::ElementwiseUnary::Type::DEQUANTIZE)
  {
    // NNAPI allow QUANT_INT8_SYMM type input
    OP_REQUIRES(isValidType(input_index, {DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_SYMM,
                                          DataType::QUANT_INT8_ASYMM}));
    OP_REQUIRES(isValidType(output_index, DataType::FLOAT32));
  }
  else if (node.param().op_type == operation::ElementwiseUnary::Type::QUANTIZE)
  {
    OP_REQUIRES(isValidType(
      input_index, {DataType::FLOAT32, DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_ASYMM}));
    OP_REQUIRES(
      isValidType(output_index, {DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_ASYMM}));
  }
  else if (node.param().op_type == operation::ElementwiseUnary::Type::FLOOR)
  {
    OP_REQUIRES(isValidType(input_index, DataType::FLOAT32));
    OP_REQUIRES(isSameType(output_index, input_index));
  }
  else if (node.param().op_type != operation::ElementwiseUnary::Type::CAST)
  {
    OP_REQUIRES(isSameType(output_index, input_index));
  }
}

void OperationValidator::visit(const operation::EmbeddingLookup &node)
{
  const auto lookups_index{node.getInputs().at(operation::EmbeddingLookup::Input::LOOKUPS)};
  const auto values_index{node.getInputs().at(operation::EmbeddingLookup::Input::VALUES)};
  const auto output_index{node.getOutputs().at(0)};

  OP_REQUIRES(isValidType(lookups_index, DataType::INT32));

  // TFLite: Allow hybrid type - value table & output
  // NNAPI: Require same value table and output type
  OP_REQUIRES(
    isSameType(values_index, output_index) ||
    (isValidType(output_index, DataType::FLOAT32) &&
     (isValidType(values_index, {DataType::QUANT_INT8_ASYMM, DataType::QUANT_INT8_SYMM}))));
}

void OperationValidator::visit(const operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(operation::ExpandDims::Input::INPUT)};
  const auto axis_index{node.getInputs().at(operation::ExpandDims::Input::AXIS)};

  OP_REQUIRES(isSameType(output_index, input_index));
  OP_REQUIRES(isValidType(axis_index, {DataType::INT32, DataType::INT64}));
}

void OperationValidator::visit(const operation::Fill &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(operation::Fill::Input::SHAPE)};
  const auto value_index{node.getInputs().at(operation::Fill::Input::VALUE)};

  OP_REQUIRES(isSameType(output_index, value_index));
  OP_REQUIRES(isValidType(input_index, {DataType::INT32, DataType::INT64}));
  OP_REQUIRES(isValidType(output_index,
                          {DataType::FLOAT32, DataType::INT32, DataType::INT64, DataType::BOOL8}));
}

void OperationValidator::visit(const operation::HashtableLookup &node)
{
  const auto hits_index{node.getOutputs().at(operation::HashtableLookup::Output::HITS)};
  const auto lookups_index{node.getInputs().at(operation::HashtableLookup::Input::LOOKUPS)};
  const auto keys_index{node.getInputs().at(operation::HashtableLookup::Input::KEYS)};

  OP_REQUIRES(isValidType(lookups_index, DataType::INT32));
  OP_REQUIRES(isValidType(keys_index, DataType::INT32));
  OP_REQUIRES(isValidType(hits_index, DataType::QUANT_UINT8_ASYMM));
}

void OperationValidator::visit(const operation::Pack &node)
{
  const auto num{node.param().num};

  OP_REQUIRES(num == static_cast<int32_t>(node.getInputs().size()));
}

void OperationValidator::visit(const operation::Pad &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(operation::Pad::Input::INPUT)};
  const auto pad_index{node.getInputs().at(operation::Pad::Input::PAD)};
  bool isQuantType =
    isValidType(output_index, {DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_ASYMM});
  bool isPadV2 = node.getInputs().size() == 3 ? true : false;

  OP_REQUIRES(isValidType(pad_index, DataType::INT32));
  OP_REQUIRES(isSameType(input_index, output_index));

  if (isQuantType)
    OP_REQUIRES(isSameQuantParam(input_index, output_index));

  if (isPadV2)
  {
    const auto value_index{node.getInputs().at(operation::Pad::Input::VALUE)};
    const bool cond_same = isSameType(input_index, value_index);
    const bool cond_same_quant = (!isQuantType || isSameQuantParam(input_index, value_index));
    const auto input_t = operandType(input_index);
    const auto value_t = operandType(value_index);
    // NNAPI accepts this case. scale and zeroPoint are assumed to be the same as in input0.
    const bool cond_quant8 =
      ((input_t == DataType::QUANT_UINT8_ASYMM || input_t == DataType::QUANT_INT8_ASYMM) &&
       value_t == DataType::INT32);
    OP_REQUIRES((cond_same && cond_same_quant) || cond_quant8);
  }
}

void OperationValidator::visit(const operation::Rank &node)
{
  const auto output_index{node.getOutputs().at(0)};

  OP_REQUIRES(isValidType(output_index, DataType::INT32));
}

void OperationValidator::visit(const operation::ResizeBilinear &node)
{
  auto align_corners = node.param().align_corners;
  auto half_pixel_centers = node.param().half_pixel_centers;

  OP_REQUIRES(!align_corners || !half_pixel_centers);
}

void OperationValidator::visit(const operation::Reverse &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(operation::Reverse::Input::INPUT)};
  const auto axis_index{node.getInputs().at(operation::Reverse::Input::AXIS)};

  OP_REQUIRES(isValidType(axis_index, DataType::INT32));
  OP_REQUIRES(isSameType(output_index, input_index));
}

void OperationValidator::visit(const operation::Select &node)
{
  const auto condition_index{node.getInputs().at(operation::Select::Input::CONDITION)};
  const auto input_true_index{node.getInputs().at(operation::Select::Input::INPUT_TRUE)};
  const auto input_false_index{node.getInputs().at(operation::Select::Input::INPUT_FALSE)};

  OP_REQUIRES(isValidType(condition_index, DataType::BOOL8));
  OP_REQUIRES(isSameType(input_true_index, input_false_index));
}

void OperationValidator::visit(const operation::Shape &node)
{
  const auto output_index{node.getOutputs().at(0)};

  OP_REQUIRES(isValidType(output_index, {DataType::UINT32, DataType::INT32, DataType::INT64}));
}

void OperationValidator::visit(const operation::Slice &node)
{
  const auto begins_index{node.getInputs().at(operation::Slice::BEGINS)};
  const auto sizes_index{node.getInputs().at(operation::Slice::SIZES)};

  OP_REQUIRES(isValidType(begins_index, {DataType::INT32, DataType::INT64}));
  OP_REQUIRES(isSameType(begins_index, sizes_index));
}

void OperationValidator::visit(const operation::Softmax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(operation::Softmax::INPUT)};

  OP_REQUIRES(isSameType(input_index, output_index));
  OP_REQUIRES(isValidType(
    output_index, {DataType::FLOAT32, DataType::QUANT_UINT8_ASYMM, DataType::QUANT_INT8_ASYMM}));
}

void OperationValidator::visit(const operation::SpaceToBatchND &node)
{
  const auto block_size_index{node.getInputs().at(operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto paddings_index{node.getInputs().at(operation::SpaceToBatchND::Input::PADDINGS)};

  // Non-constant block_size and padding is not implemented yet
  OP_REQUIRES(isConstant(block_size_index));
  OP_REQUIRES(isConstant(paddings_index));
}

void OperationValidator::visit(const operation::SpaceToDepth &node)
{
  const auto block_size = node.param().block_size;
  OP_REQUIRES(block_size >= 1);
}

void OperationValidator::visit(const operation::Split &node)
{
  const auto num_splits = node.param().num_splits;

  OP_REQUIRES(num_splits > 0 && num_splits <= 0xFFFF);
  OP_REQUIRES(node.getOutputs().size() == static_cast<uint32_t>(num_splits));
}

void OperationValidator::visit(const operation::SquaredDifference &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(operation::SquaredDifference::Input::LHS)};
  const auto rhs_index{node.getInputs().at(operation::SquaredDifference::Input::RHS)};

  OP_REQUIRES(isSameType(output_index, lhs_index));
  OP_REQUIRES(isSameType(lhs_index, rhs_index));
}

void OperationValidator::visit(const operation::StatelessRandomUniform &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto shape_index{node.getInputs().at(operation::StatelessRandomUniform::Input::SHAPE)};
  const auto seed_index{node.getInputs().at(operation::StatelessRandomUniform::Input::SEED)};

  OP_REQUIRES(isValidType(output_index, DataType::FLOAT32));
  OP_REQUIRES(isValidType(shape_index, DataType::INT32));
  OP_REQUIRES(isValidType(seed_index, DataType::INT32));
}

void OperationValidator::visit(const operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(operation::StridedSlice::Input::INPUT)};

  OP_REQUIRES(isSameType(output_index, input_index));
}

void OperationValidator::visit(const operation::TransposeConv &node)
{
  OP_REQUIRES((node.param().padding.type == PaddingType::SAME) ||
              (node.param().padding.type == PaddingType::VALID));
}

void OperationValidator::visit(const operation::Unpack &node)
{
  const auto num{node.param().num};
  OP_REQUIRES(num == static_cast<int32_t>(node.getOutputs().size()));
}

void OperationValidator::visit(const operation::While &node)
{
  OP_REQUIRES(node.getInputs().size() == node.getOutputs().size());
}

} // namespace ir
} // namespace onert
