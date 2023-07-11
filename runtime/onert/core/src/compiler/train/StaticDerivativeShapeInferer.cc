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

#include "StaticDerivativeShapeInferer.h"
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

void StaticDerivativeShapeInferer::infer()
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
      msg << "StaticDerivativeShapeInferer does not support dynamic shape yet, ";
      msg << op.name() << "(op index: " << op_idx << ") has dynamic shape.";
      throw std::runtime_error(msg.str());
    }
    else
      op.accept(*this);
  }
}

void StaticDerivativeShapeInferer::dump()
{
  // TODO dump
}

bool StaticDerivativeShapeInferer::checkDynamicInput(const ir::IOperation &op)
{
  const auto &operands = _lowered_subg->graph().operands();
  for (auto input_idx : op.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
  {
    if (operands.at(input_idx).info().isDynamic())
    {
      return true;
    }
  }

  return false;
}

void StaticDerivativeShapeInferer::visit(const ir::train::operation::Conv2D &)
{
  // NYI
}

void StaticDerivativeShapeInferer::visit(const ir::train::operation::ElementwiseActivation &)
{
  // NYI
}

void StaticDerivativeShapeInferer::visit(const ir::train::operation::Loss &)
{
  // NYI
}

void StaticDerivativeShapeInferer::visit(const ir::train::operation::Permute &)
{
  // NYI
}

void StaticDerivativeShapeInferer::visit(const ir::train::operation::Pool2D &)
{
  // NYI
}

void StaticDerivativeShapeInferer::visit(const ir::train::operation::Reshape &)
{
  // NYI
}

void StaticDerivativeShapeInferer::visit(const ir::train::operation::Softmax &)
{
  // NYI
}

} // namespace train
} // namespace compiler
} // namespace onert
