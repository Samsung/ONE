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

OperationValidator::OperationValidator(const ir::Graph &graph)
    : _graph{graph}, _ctx{graph.operands()}
{
}

void OperationValidator::operator()()
{
  assert(_graph.subgraphs() == nullptr);

  _graph.operations().iterate(
      [&](const ir::OperationIndex &, const ir::Operation &node) { node.accept(*this); });
}

ir::DataType OperationValidator::operandType(const ir::OperandIndex &idx)
{
  return _graph.operands().at(idx).typeInfo().type();
}

bool OperationValidator::isConstant(const ir::OperandIndex &idx)
{
  return _graph.operands().at(idx).isConstant();
}

bool OperationValidator::isSameType(const ir::OperandIndex &idx1, const ir::OperandIndex &idx2)
{
  return operandType(idx1) == operandType(idx2);
}

bool OperationValidator::isValidType(const ir::OperandIndex &idx, const ir::DataType &type)
{
  return operandType(idx) == type;
}

bool OperationValidator::isValidType(const ir::OperandIndex &idx,
                                     std::initializer_list<ir::DataType> valid_types)
{
  for (auto type_to_check : valid_types)
  {
    if (isValidType(idx, type_to_check))
    {
      return true;
    }
  }

  return false;
}

void OperationValidator::visit(const ir::operation::AddN &node)
{
  int size = node.getInputs().size();
  for (int i = 0; i < size; i++)
  {
    const auto input_index(node.getInputs().at(i));
    OP_REQUIRES(isValidType(input_index, {ir::DataType::FLOAT32, ir::DataType::INT32}));
  }
}

void OperationValidator::visit(const ir::operation::BatchMatMul &node)
{
  const auto lhs_index(node.getInputs().at(ir::operation::BatchMatMul::Input::LHS));
  const auto rhs_index(node.getInputs().at(ir::operation::BatchMatMul::Input::RHS));

  // Constant lhs and rhs is not implemented yet
  OP_REQUIRES(!isConstant(lhs_index) && !isConstant(rhs_index));
}

void OperationValidator::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto block_size_index{
      node.getInputs().at(ir::operation::BatchToSpaceND::Input::BLOCK_SIZE)};

  // Non-constant block_size is not implemented yet
  OP_REQUIRES(isConstant(block_size_index));
}

void OperationValidator::visit(const ir::operation::BinaryArithmetic &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS)};

  OP_REQUIRES(isSameType(lhs_index, rhs_index));
  OP_REQUIRES(isSameType(lhs_index, output_index));
}

void OperationValidator::visit(const ir::operation::Comparison &node)
{
  const auto output_index{node.getOutputs().at(0)};

  const auto lhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  OP_REQUIRES(isSameType(lhs_index, rhs_index));
  OP_REQUIRES(isValidType(output_index, ir::DataType::BOOL8));
}

void OperationValidator::visit(const ir::operation::DepthToSpace &node)
{
  int32_t block_size = node.param().block_size;

  OP_REQUIRES(block_size > 0);
}

void OperationValidator::visit(const ir::operation::ElementwiseActivation &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  // Check if I/O types match
  OP_REQUIRES(isSameType(output_index, input_index));
}

void OperationValidator::visit(const ir::operation::ElementwiseBinary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::RHS)};

  OP_REQUIRES(isSameType(lhs_index, rhs_index));
  OP_REQUIRES(isSameType(lhs_index, output_index));
}

void OperationValidator::visit(const ir::operation::ElementwiseUnary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ElementwiseUnary::Input::INPUT)};

  // Check if I/O types match
  if (node.param().op_type == ir::operation::ElementwiseUnary::Type::DEQUANTIZE)
  {
    // NNAPI allow QUANT_INT8_SYMM type input
    OP_REQUIRES(
        isValidType(input_index, {ir::DataType::QUANT_UINT8_ASYMM, ir::DataType::QUANT_INT8_SYMM,
                                  ir::DataType::QUANT_INT8_ASYMM}));
    OP_REQUIRES(isValidType(output_index, ir::DataType::FLOAT32));
  }
  else if (node.param().op_type == ir::operation::ElementwiseUnary::Type::QUANTIZE)
  {
    OP_REQUIRES(isValidType(input_index, ir::DataType::FLOAT32));
    OP_REQUIRES(isValidType(output_index, ir::DataType::QUANT_UINT8_ASYMM));
  }
  else if (node.param().op_type != ir::operation::ElementwiseUnary::Type::CAST)
  {
    OP_REQUIRES(isSameType(output_index, input_index));
  }
}

void OperationValidator::visit(const ir::operation::EmbeddingLookup &node)
{
  const auto lookups_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::LOOKUPS)};

  OP_REQUIRES(isValidType(lookups_index, ir::DataType::INT32));
}

void OperationValidator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};

  OP_REQUIRES(isSameType(output_index, input_index));
  OP_REQUIRES(isValidType(axis_index, ir::DataType::INT32));
}

void OperationValidator::visit(const ir::operation::HashtableLookup &node)
{
  const auto hits_index{node.getOutputs().at(ir::operation::HashtableLookup::Output::HITS)};
  const auto lookups_index{node.getInputs().at(ir::operation::HashtableLookup::Input::LOOKUPS)};
  const auto keys_index{node.getInputs().at(ir::operation::HashtableLookup::Input::KEYS)};

  OP_REQUIRES(isValidType(lookups_index, ir::DataType::INT32));
  OP_REQUIRES(isValidType(keys_index, ir::DataType::INT32));
  OP_REQUIRES(isValidType(hits_index, ir::DataType::QUANT_UINT8_ASYMM));
}

void OperationValidator::visit(const ir::operation::Pack &node)
{
  const auto num{node.param().num};

  OP_REQUIRES(num == static_cast<int32_t>(node.getInputs().size()));
}

void OperationValidator::visit(const ir::operation::Pad &node)
{
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};

  OP_REQUIRES(isValidType(pad_index, ir::DataType::INT32));
}

void OperationValidator::visit(const ir::operation::ResizeBilinear &node)
{
  auto align_corners = node.param().align_corners;
  auto half_pixel_centers = node.param().half_pixel_centers;

  OP_REQUIRES(!align_corners || !half_pixel_centers);
}

void OperationValidator::visit(const ir::operation::Reverse &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reverse::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::Reverse::Input::AXIS)};

  OP_REQUIRES(isValidType(axis_index, ir::DataType::INT32));
  OP_REQUIRES(isSameType(output_index, input_index));
}

void OperationValidator::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto block_size_index{
      node.getInputs().at(ir::operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto paddings_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::PADDINGS)};

  // Non-constant block_size and padding is not implemented yet
  OP_REQUIRES(isConstant(block_size_index));
  OP_REQUIRES(isConstant(paddings_index));
}

void OperationValidator::visit(const ir::operation::SpaceToDepth &node)
{
  const auto block_size = node.param().block_size;
  OP_REQUIRES(block_size >= 1);
}

void OperationValidator::visit(const ir::operation::Select &node)
{
  const auto condition_index{node.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto input_true_index{node.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto input_false_index{node.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};

  OP_REQUIRES(isValidType(condition_index, ir::DataType::BOOL8));
  OP_REQUIRES(isSameType(input_true_index, input_false_index));
}

void OperationValidator::visit(const ir::operation::Split &node)
{
  const auto num_splits = node.param().num_splits;

  OP_REQUIRES(num_splits > 0 && num_splits <= 0xFFFF);
  OP_REQUIRES(node.getOutputs().size() == static_cast<uint32_t>(num_splits));
}

void OperationValidator::visit(const ir::operation::SquaredDifference &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::RHS)};

  OP_REQUIRES(isSameType(output_index, lhs_index));
  OP_REQUIRES(isSameType(lhs_index, rhs_index));
}

void OperationValidator::visit(const ir::operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};

  OP_REQUIRES(isSameType(output_index, input_index));
}

void OperationValidator::visit(const ir::operation::TransposeConv &node)
{
  OP_REQUIRES((node.param().padding.type == ir::PaddingType::SAME) ||
              (node.param().padding.type == ir::PaddingType::VALID));
}

void OperationValidator::visit(const ir::operation::Unpack &node)
{
  const auto num{node.param().num};
  OP_REQUIRES(num == static_cast<int32_t>(node.getOutputs().size()));
}

void OperationValidator::visit(const ir::operation::While &node)
{
  OP_REQUIRES(node.getInputs().size() == node.getOutputs().size());
}

} // namespace compiler
} // namespace onert
