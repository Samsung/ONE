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

#include <loco.h>

namespace luci
{

bool CircleCustomGraphBuilder::validate(const ValidateArgs &) const
{
  // DO NOTHING
  return true;
}

CircleNode *CircleCustomGraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  uint32_t input_count = bna.op.inputs.size();
  uint32_t output_count = bna.op.outputs.size();

  auto *node = bna.context->graph()->nodes()->create<CircleCustom>(input_count, output_count);

  for (uint32_t idx = 0; idx < input_count; ++idx)
  {
    node->inputs(idx, bna.input_nodes[idx]);
  }

  const auto &opcodes = bna.context->reader()->native_opcodes();
  const uint32_t opcode_index = bna.op.opcode_index;
  const auto opcode = opcodes[opcode_index];
  assert(opcode != nullptr);

  node->custom_options(
    std::vector<uint8_t>{bna.op.custom_options.begin(), bna.op.custom_options.end()});

  assert(opcode->custom_code() != nullptr);
  node->custom_code(opcode->custom_code()->c_str());

  // NOTE Operator version of custom is always 1

  return node;
}

CircleNode *CircleCustomGraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleCustomOut>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
