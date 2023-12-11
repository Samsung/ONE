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

#include "ConstantLoweringPass.h"

#include "backend/Backend.h"
#include <ir/Graph.h>
#include <compiler/PermuteFactor.h>
#include <util/Utils.h>
#include "util/logging.h"

namespace onert
{
namespace compiler
{
namespace pass
{

void ConstantLoweringPass::callback(const ir::OperationIndex &node_index, ir::IOperation &node)
{
  const auto op_lower_info = _lowered_graph.lower_info().operation.getRawPtr(node_index);
  const auto backend = op_lower_info->backend();
  const auto factor = PermuteFactor{backend};

  // Now this runtime does not support the node making output of operation as constant
  for (const auto &input : node.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
  {
    auto &object = _graph.operands().at(input);
    if (object.isConstant())
    {
      // All constant operand are already assinged at each backend by ContantInsertionPass. So a
      // constant has `def` and `use` as the same PermuteFactor
      auto operand_li = std::make_unique<compiler::OperandLowerInfo>();
      operand_li->addDefPermuteFactor(factor);
      operand_li->addUsePermuteFactor(factor);
      _lowered_graph.lower_info().operand.set(input, std::move(operand_li));
    }
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
