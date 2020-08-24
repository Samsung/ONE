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
#include <ir/operand/PermuteFactor.h>
#include <util/Utils.h>

namespace onert
{
namespace compiler
{
namespace pass
{

void ConstantLoweringPass::callback(const ir::OperationIndex &node_index, ir::Operation &node)
{
  const auto &op_sequence_index = _lowered_graph.op_seqs().getOperation(node_index);
  const auto op_seq_lower_info = _lowered_graph.getLowerInfo(op_sequence_index);
  const auto backend = op_seq_lower_info->backend();
  const auto layout = op_seq_lower_info->layout();
  const auto factor = ir::operand::PermuteFactor{backend, layout};

  // Now this runtime does not support the node making output of operation as constant
  for (const auto input : node.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
  {
    auto &object = _graph.operands().at(input);
    if (object.isConstant())
    {
      // All constant operand are already assinged at each backend by ContantInsertionPass. So a
      // constant has `def` and `use` as the same PermuteFactor
      _lowered_graph.setLowerInfo(input, std::make_unique<ir::operand::LowerInfo>());
      _lowered_graph.getLowerInfo(input)->addDefPermuteFactor(factor);
      _lowered_graph.getLowerInfo(input)->addUsePermuteFactor(factor);
    }
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
