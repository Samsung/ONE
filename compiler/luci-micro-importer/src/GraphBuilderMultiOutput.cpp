/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/GraphBuilderMultiOutput.h"

#include <luci/Log.h>

namespace luci
{

CircleNode *GraphBuilderMultiOutput::build(const circle::OperatorT &op,
                                           GraphBuilderContext *context) const
{
  LOGGER(l);

  assert(context != nullptr);

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto tensors = context->reader()->tensors();
  const auto &opcodes = context->reader()->opcodes();
  assert(not tensors.is_null());

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    if (input_tensor_index >= 0)
    {
      auto input = context->nodefinder()->node(input_tensor_index);
      if (input == nullptr)
        INFO(l) << "[luci] Warning: input node is null " << input_tensor_index << std::endl;
      input_nodes.push_back(input);
    }
    else
    {
      // If there is no tensor, insert CircleOutputExclude.
      auto *node = context->graph()->nodes()->create<luci::CircleOutputExclude>();
      // CircleOutputExclude doesn't need a type, but since all nodes must have a type,
      // a dummy type is inserted.
      node->dtype(loco::DataType::FLOAT32);
      input_nodes.push_back(node);
    }
  }

  BuildNodeArgs bna(op, context, input_nodes);
  auto *node = build_node(bna);

  uint32_t output_count = outputs.size();
  // NOTE CustomOp inherits GraphBuilderMultiOutput and can have 0 output
  if (output_count > 0)
  {
    // Let's use attributes from output 0 for this node
    auto const output_tensor = tensors[outputs[0]];
    assert(output_tensor != nullptr);

    node->name(tensor_name(output_tensor));
    node->dtype(luci_datatype(output_tensor->type()));

    // mark operator version
    assert(opcodes[op.opcode_index] != nullptr);
    node->op_version(opcodes[op.opcode_index]->version());

    // NOTE We don't set quantization for multiple output nodes but to virtual outputs
  }

  // Create virtual outputs of Virtual Output node(s)
  for (uint32_t n = 0; n < output_count; ++n)
  {
    auto const output_tensor = tensors[outputs[n]];
    assert(output_tensor != nullptr);

    BuildOutArgs boa(node, n);
    auto *nodeout = build_out(boa);

    copy_tensor_attributes(output_tensor, nodeout);
    // NOTE name of CxxxOut nodes may have same name
    // mark shape_status
    if (output_tensor->shape() == nullptr)
      nodeout->shape_status(ShapeStatus::NOSHAPE);
    else
      nodeout->shape_status(ShapeStatus::VALID);

    context->nodefinder()->enroll(outputs[n], nodeout);
  }

  return node;
}

} // namespace luci
