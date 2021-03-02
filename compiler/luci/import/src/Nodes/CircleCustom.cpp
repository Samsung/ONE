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

#if 0
CircleNode *CircleCustomGraphBuilder::build(const circle::OperatorT &op,
                                            GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  auto tensors_ptr = context->reader()->tensors_ptr();
  assert(tensors_ptr != nullptr);

  // Create CircleCustom
  const auto &opcodes = context->reader()->opcodes();
  const uint32_t opcode_index = op.opcode_index;
  const circle::OperatorCodeT &opcode = *opcodes[opcode_index];

  auto *node = graph->nodes()->create<CircleCustom>(inputs.size(), outputs.size());
  uint32_t input_idx = 0;
  for (const int32_t input_tensor_index : inputs)
  {
    node->inputs(input_idx++, context->nodefinder()->node(input_tensor_index));
  }
  node->custom_options(std::vector<uint8_t>{op.custom_options.begin(), op.custom_options.end()});
  node->custom_code(opcode.custom_code);
  // Operator version of custom is always 1, so do nothing

  uint32_t output_count = outputs.size();

  assert(output_count > 0);
  {
    // Let's use attributes from output 0 for this node
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    node->name(tensor_name(output_tensor));
    node->dtype(luci_datatype(output_tensor.type));
  }

  // Create virtual outputs of Custom
  for (uint32_t n = 0; n < output_count; ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleCustomOut>();
    copy_tensor_attributes(output_tensor, nodeout);
    // mark shape_status
    if (tensors_ptr->Get(outputs[n])->shape() == nullptr)
      nodeout->shape_status(ShapeStatus::NOSHAPE);
    else
      nodeout->shape_status(ShapeStatus::VALID);

    nodeout->input(node);
    nodeout->index(n);

    context->nodefinder()->enroll(outputs[n], nodeout);
  }

  return node;
}
#endif

CircleNode *CircleCustomGraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  uint32_t input_count = bna.op.inputs.size();
  uint32_t output_count = bna.op.outputs.size();

  auto *node = bna.context->graph()->nodes()->create<CircleCustom>(input_count, output_count);

  for (uint32_t idx = 0; idx < input_count; ++idx)
  {
    node->inputs(idx, bna.input_nodes[idx]);
  }

  const auto &opcodes = bna.context->reader()->opcodes();
  const uint32_t opcode_index = bna.op.opcode_index;
  const circle::OperatorCodeT &opcode = *opcodes[opcode_index];

  node->custom_options(
    std::vector<uint8_t>{bna.op.custom_options.begin(), bna.op.custom_options.end()});
  node->custom_code(opcode.custom_code);

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
