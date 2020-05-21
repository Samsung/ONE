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

void CircleCustomGraphBuilder::build(const circle::OperatorT &op,
                                     GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();

  // Create CircleCustom
  const auto &opcodes = context->reader()->opcodes();
  const uint32_t opcode_index = op.opcode_index;
  const circle::OperatorCodeT &opcode = *opcodes[opcode_index];

  auto *node = graph->nodes()->create<CircleCustom>(inputs.size());
  uint32_t input_idx = 0;
  for (const int32_t input_tensor_index : inputs)
  {
    node->inputs(input_idx++, context->nodefinder()->node(input_tensor_index));
  }
  node->custom_options(std::vector<uint8_t>{op.custom_options.begin(), op.custom_options.end()});
  node->custom_code(opcode.custom_code);

  assert(outputs.size() == 1);
  {
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    copy_tensor_attributes(output_tensor, node);
  }

  context->nodefinder()->enroll(outputs[0], node);
}

} // namespace luci
