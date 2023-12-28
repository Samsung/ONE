/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "StaticBackwardShapeInferer.h"
#include "util/ShapeInference.h"
#include "util/logging.h"

#include <misc/polymorphic_downcast.h>

#include <sstream>
#include <stdexcept>

namespace onert
{
namespace compiler
{
namespace train
{

void StaticBackwardShapeInferer::infer()
{
  // It is not determined to iterate in reverse order.
  auto sorted_ops = _lowered_subg->graph().topolSortOperations();
  for (auto it = sorted_ops.rbegin(); it != sorted_ops.rend(); ++it)
  {
    const auto op_idx = *it;
    const auto &op = _lowered_subg->trainable_graph().operation(op_idx);
    if (checkDynamicInput(op))
    {
      std::stringstream msg;
      msg << "StaticBackwardShapeInferer does not support dynamic shape yet, ";
      msg << op.name() << "(op index: " << op_idx << ") has dynamic shape.";
      throw std::runtime_error(msg.str());
    }

    checkOutput(op);

    op.accept(*this);
  }
}

void StaticBackwardShapeInferer::dump()
{
  // TODO dump
}

bool StaticBackwardShapeInferer::checkDynamicInput(const ir::IOperation &op)
{
  const auto &operands = _lowered_subg->graph().operands();
  for (const auto &input_idx : op.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
  {
    if (operands.at(input_idx).info().isDynamic())
    {
      return true;
    }
  }

  return false;
}

void StaticBackwardShapeInferer::checkOutput(const ir::IOperation &op)
{
  const auto &bwd_operands = _lowered_subg->trainable_graph().backward_operands();
  for (const auto &output_idx : op.getOutputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
  {
    if (!bwd_operands.exist(output_idx))
    {
      std::stringstream msg;
      msg << "StaticBackwardShapeInferer : Invalid output, ";
      msg << op.name() << "'s back propagation output(index: " << output_idx << ") does not exist.";
      throw std::runtime_error(msg.str());
    }
  }
}

void StaticBackwardShapeInferer::setShape(const ir::OperandIndex &index, const ir::Shape &shape)
{
  auto &tgraph = _lowered_subg->trainable_graph();

  if (tgraph.backward_operands().exist(index))
    tgraph.changeBackwardShape(index, shape);
  else
  {
    // NOTE This code assumes the types are always the same, but I'm not sure.
    const auto &type = tgraph.operands().at(index).typeInfo();
    const auto new_index =
      tgraph.addBackwardOperand(index, std::make_unique<ir::Operand>(shape, type));
    assert(new_index == index);
    UNUSED_RELEASE(new_index);
  }
}

void StaticBackwardShapeInferer::visit(const ir::train::operation::Conv2D &)
{
  // NYI
}

void StaticBackwardShapeInferer::visit(const ir::train::operation::ElementwiseActivation &)
{
  // NYI
}

void StaticBackwardShapeInferer::visit(const ir::train::operation::Loss &)
{
  // NYI
}

void StaticBackwardShapeInferer::visit(const ir::train::operation::Permute &op)
{
  const auto &bwd_operands = _lowered_subg->trainable_graph().backward_operands();

  const auto &output_idx = op.getOutputs().at(0);
  const auto &output = bwd_operands.at(output_idx);

  // re-sizing shape of back propagatation input
  const auto &input_idx = op.getInputs().at(0);
  const auto &new_shape = output.info().shape();
  setShape(input_idx, new_shape);
}

void StaticBackwardShapeInferer::visit(const ir::train::operation::Pool2D &)
{
  // NYI
}

void StaticBackwardShapeInferer::visit(const ir::train::operation::Reshape &)
{
  // NYI
}

void StaticBackwardShapeInferer::visit(const ir::train::operation::Softmax &)
{
  // NYI
}

} // namespace train
} // namespace compiler
} // namespace onert
