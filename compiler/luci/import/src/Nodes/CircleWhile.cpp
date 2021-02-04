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

#include "luci/Import/Nodes/CircleWhile.h"

#include <luci/IR/Nodes/CircleWhile.h>
#include <luci/IR/Nodes/CircleWhileOut.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleWhileGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto *options = args.op.builtin_options.AsWhileOptions();

  if (inputs.size() != args.op.outputs.size())
    return false;

  auto num_graphs = static_cast<int32_t>(args.reader.num_subgraph());
  if (options->cond_subgraph_index >= num_graphs)
    return false;
  if (options->body_subgraph_index >= num_graphs)
    return false;

  return true;
}

/**
 * @brief  While Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple CircleWhileOut nodes to emulate this
 *         For two outputs that may look like this
 *
 *         --- CircleWhile --- Node ---
 *                       \- Node ---
 *
 *         will be created like this
 *
 *         --- CircleWhile --- CircleWhileOut --- Node ---
 *                       \- CircleWhileOut --- Node ---
 */

CircleNode *CircleWhileGraphBuilder::build(const circle::OperatorT &op,
                                           GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  const auto &opcodes = context->reader()->opcodes();

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    auto input_node = context->nodefinder()->node(input_tensor_index);
    assert(input_node != nullptr);
    input_nodes.push_back(input_node);
  }

  uint32_t input_count = inputs.size();
  uint32_t output_count = outputs.size();

  // Create CircleWhile
  CircleWhile *node = graph->nodes()->create<CircleWhile>(input_count, output_count);

  for (uint32_t idx = 0; idx < input_count; ++idx)
  {
    node->input(idx, input_nodes[idx]);
  }

  const auto *options = op.builtin_options.AsWhileOptions();
  node->cond_branch(options->cond_subgraph_index);
  node->body_branch(options->body_subgraph_index);

  assert(outputs.size() > 0);
  {
    // Lets use name of output 0 as While name
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    node->name(tensor_name(output_tensor));
    node->op_version(opcodes[op.opcode_index].get()->version);

    // NOTE We don't set quantization for While itself but to virtual outputs
  }

  // Create virtual outputs of While
  for (uint32_t n = 0; n < output_count; ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleWhileOut>();

    nodeout->input(node);
    nodeout->index(n);

    copy_tensor_attributes(output_tensor, nodeout);

    // Note: leave shape_status to UNKNOWN to run shape inference

    context->nodefinder()->enroll(outputs[n], nodeout);
  }

  return node;
}

} // namespace luci
