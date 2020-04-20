/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleCustom.h"

#include <luci/IR/Nodes/CircleBatchMatMul.h>

#include <loco.h>

namespace luci
{

bool CircleCustomGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &opcodes = args.reader.opcodes();
  const uint32_t opcode_index = args.op.opcode_index;
  const circle::OperatorCodeT &opcode = *opcodes[opcode_index];

  if (opcode.builtin_code != circle::BuiltinOperator_CUSTOM)
    return false;

  return true;
}

CircleNode *CircleCustomGraphBuilder::build_node(const circle::OperatorT &op,
                                                 const std::vector<CircleNode *> &inputs,
                                                 loco::Graph *graph, CircleReader *reader) const
{
  const auto &opcodes = reader->opcodes();
  const uint32_t opcode_index = op.opcode_index;
  const circle::OperatorCodeT &opcode = *opcodes[opcode_index];

  CircleNode *node;
  if (opcode.custom_code == "BatchMatMulV2")
  {
    auto *custom_node = graph->nodes()->create<CircleBatchMatMul>();
    custom_node->x(inputs[0]);
    custom_node->y(inputs[1]);

    const auto *options = op.builtin_options.AsBatchMatMulOptions();
    custom_node->adj_x(options->adjoint_lhs);
    custom_node->adj_y(options->adjoint_rhs);
    node = custom_node;
  }

  return node;
}

} // namespace luci
