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
namespace compiler
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

void OperationValidator::visit(const ir::operation::BatchMatMul &node)
{
  const auto lhs_index(node.getInputs().at(ir::operation::BatchMatMul::Input::LHS));
  const auto rhs_index(node.getInputs().at(ir::operation::BatchMatMul::Input::RHS));

  // Constant lhs and rhs is not implemented yet
  OP_REQUIRES(!_ctx.at(lhs_index).isConstant() && !_ctx.at(rhs_index).isConstant());
}

void OperationValidator::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto block_size_index{
      node.getInputs().at(ir::operation::BatchToSpaceND::Input::BLOCK_SIZE)};

  // Non-constant block_size is not implemented yet
  OP_REQUIRES(_ctx.at(block_size_index).isConstant());
}

void OperationValidator::visit(const ir::operation::BinaryArithmetic &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS)};

  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(output_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::Comparison &node)
{
  const auto output_index{node.getOutputs().at(0)};

  const auto lhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == ir::DataType::BOOL8);
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
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::ElementwiseBinary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::RHS)};

  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(output_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::ElementwiseUnary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ElementwiseUnary::Input::INPUT)};

  // Check if I/O types match
  if (node.param().op_type == ir::operation::ElementwiseUnary::Type::DEQUANTIZE)
  {
    OP_REQUIRES((_ctx.at(input_index).typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM) ||
                (_ctx.at(input_index).typeInfo().type() == ir::DataType::QUANT_INT8_SYMM));
    OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == ir::DataType::FLOAT32);
  }
  else if (node.param().op_type == ir::operation::ElementwiseUnary::Type::QUANTIZE)
  {
    OP_REQUIRES(_ctx.at(input_index).typeInfo().type() == ir::DataType::FLOAT32);
    OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM);
  }
  else if (node.param().op_type != ir::operation::ElementwiseUnary::Type::CAST)
  {
    OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
  }
}

void OperationValidator::visit(const ir::operation::EmbeddingLookup &node)
{
  const auto lookups_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::LOOKUPS)};

  OP_REQUIRES(_ctx.at(lookups_index).typeInfo().type() == ir::DataType::INT32);
}

void OperationValidator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};

  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(axis_index).typeInfo().type() == ir::DataType::INT32);
}

void OperationValidator::visit(const ir::operation::HashtableLookup &node)
{
  const auto hits_index{node.getOutputs().at(ir::operation::HashtableLookup::Output::HITS)};
  const auto lookups_index{node.getInputs().at(ir::operation::HashtableLookup::Input::LOOKUPS)};
  const auto keys_index{node.getInputs().at(ir::operation::HashtableLookup::Input::KEYS)};

  OP_REQUIRES(_ctx.at(lookups_index).typeInfo().type() == ir::DataType::INT32);
  OP_REQUIRES(_ctx.at(keys_index).typeInfo().type() == ir::DataType::INT32);
  OP_REQUIRES(_ctx.at(hits_index).typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM);
}

void OperationValidator::visit(const ir::operation::Pack &node)
{
  const auto num{node.param().num};

  OP_REQUIRES(num == static_cast<int32_t>(node.getInputs().size()));
}

void OperationValidator::visit(const ir::operation::Pad &node)
{
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};

  OP_REQUIRES(_ctx.at(pad_index).typeInfo().type() == ir::DataType::INT32);
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

  OP_REQUIRES(_ctx.at(axis_index).typeInfo().type() == ir::DataType::INT32);
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto block_size_index{
      node.getInputs().at(ir::operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto paddings_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::PADDINGS)};

  // Non-constant block_size and padding is not implemented yet
  OP_REQUIRES(_ctx.at(block_size_index).isConstant());
  OP_REQUIRES(_ctx.at(paddings_index).isConstant());
}

void OperationValidator::visit(const ir::operation::SpaceToDepth &node)
{
  const auto block_size = node.param().block_size;
  OP_REQUIRES(block_size >= 1);
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

  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(lhs_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};

  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
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
