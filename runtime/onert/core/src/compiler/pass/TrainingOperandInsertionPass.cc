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

#include "TrainingOperandInsertionPass.h"

#include "ir/Graph.h"
#include "ir/Operations.Include.h"

#include <cassert>
#include <string>

namespace onert
{
namespace compiler
{
namespace pass
{

void TrainingOperandInsertionPass::callback(const ir::OperationIndex &, ir::Operation &node)
{
  node.accept(*this);
}

void TrainingOperandInsertionPass::visit(ir::operation::ElementwiseActivation &node)
{
  assert(node.training_indices().size() == 0);

  if (node.param().op_type == ir::operation::ElementwiseActivation::Type::RELU)
  {
    const auto &output_ind = node.getOutputs().at(0);
    const auto &output_obj = _graph.operands().at(output_ind);
    const auto &output_shape = output_obj.shape();
    const auto &output_type = output_obj.typeInfo();

    auto flex_ind = _graph.addOperand(output_shape, output_type);
    node.training_indices().emplace_back(flex_ind);
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
