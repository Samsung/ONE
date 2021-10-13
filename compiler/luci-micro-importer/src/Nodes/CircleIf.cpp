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
  const auto &tensors = args.reader.native_tensors();
  const auto &tensor = tensors.at(inputs.at(0));
  assert(tensor != nullptr);
  if (tensor->type() != circle::TensorType_BOOL)
    return false;

  const auto &shape = wrap(tensor->shape());
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

CircleNode *CircleIfGraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  uint32_t input_count = bna.op.inputs.size() - 1;
  uint32_t output_count = bna.op.outputs.size();

  auto *node = bna.context->graph()->nodes()->create<CircleIf>(input_count, output_count);

  node->cond(bna.input_nodes[0]);
  for (uint32_t idx = 0; idx < input_count; ++idx)
  {
    node->input(idx, bna.input_nodes[idx + 1]);
  }

  const auto *options = bna.op.builtin_options.AsIfOptions();
  node->then_branch(options->then_subgraph_index);
  node->else_branch(options->else_subgraph_index);

  return node;
}

CircleNode *CircleIfGraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleIfOut>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
