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

#include "luci/Import/Nodes/CircleIf.h"

#include <luci/IR/Nodes/CircleIf.h>
#include <luci/IR/Nodes/CircleIfOut.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleIfGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto *options = args.op.builtin_options.AsIfOptions();

  if (inputs.size() < 2) // cond + input
    return false;
  if (args.op.outputs.size() < 1) // output
    return false;

  auto num_graphs = static_cast<int32_t>(args.reader.num_subgraph());
  if (options->then_subgraph_index >= num_graphs)
    return false;
  if (options->else_subgraph_index >= num_graphs)
    return false;

  // input 0 should be BOOL type
  const auto &tensors = args.reader.tensors();
  const auto &tensor = tensors.at(inputs[0]);
  if (tensor->type != circle::TensorType_BOOL)
    return false;

  const auto &shape = tensor->shape;
  if (shape.size() != 1 && shape.size() != 0)
    return false;

  return true;
}

/**
 * @brief  If Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple CircleIfOut nodes to emulate this
 *         For two outputs that may look like this
 *
 *         --- CircleIf --- Node ---
 *                       \- Node ---
 *
 *         will be created like this
 *
 *         --- CircleIf --- CircleIfOut --- Node ---
 *                       \- CircleIfOut --- Node ---
 */

void CircleIfGraphBuilder::build(const circle::OperatorT &op, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  const auto &opcodes = context->reader()->opcodes();
  auto tensors_ptr = context->reader()->tensors_ptr();
  assert(tensors_ptr != nullptr);

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    input_nodes.push_back(context->nodefinder()->node(input_tensor_index));
  }

  uint32_t input_count = inputs.size() - 1;
  uint32_t output_count = outputs.size();

  // Create CircleIf
  CircleIf *node = graph->nodes()->create<CircleIf>(input_count, output_count);

  node->cond(input_nodes[0]);
  for (uint32_t idx = 0; idx < input_count; ++idx)
  {
    node->input(idx, input_nodes[idx + 1]);
  }

  const auto *options = op.builtin_options.AsIfOptions();
  node->then_branch(options->then_subgraph_index);
  node->else_branch(options->else_subgraph_index);

  assert(outputs.size() > 0);
  {
    // Lets use name of output 0 as If name
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    node->name(tensor_name(output_tensor));
    node->op_version(opcodes[op.opcode_index].get()->version);

    // NOTE We don't set quantization for If itself but to virtual outputs
  }

  // Create virtual outputs of If
  for (uint32_t n = 0; n < output_count; ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleIfOut>();
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
}

} // namespace luci
