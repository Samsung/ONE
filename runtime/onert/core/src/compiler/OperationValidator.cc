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

void OperationValidator::visit(const ir::operation::Comparison &node)
{
  const auto output_index{node.getOutputs().at(0)};

  const auto lhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == ir::DataType::BOOL8);
}

void OperationValidator::visit(const ir::operation::ElementwiseActivation &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  // Check if I/O types match
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

} // namespace compiler
} // namespace onert
