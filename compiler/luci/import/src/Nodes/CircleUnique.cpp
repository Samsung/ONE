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

#include "luci/Import/Nodes/CircleUnique.h"

#include <luci/IR/Nodes/CircleUnique.h>
#include <luci/IR/Nodes/CircleUniqueOut.h>

#include <loco.h>

namespace luci
{

bool CircleUniqueGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 1)
    return false;

  if (args.op.outputs.size() != 2)
    return false;

  return true;
}

#if 0
CircleNode *CircleUniqueGraphBuilder::build(const circle::OperatorT &op,
                                            GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  auto tensors_ptr = context->reader()->tensors_ptr();
  assert(tensors_ptr != nullptr);

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    input_nodes.push_back(context->nodefinder()->node(input_tensor_index));
  }

  // Create CircleUnique
  auto node = graph->nodes()->create<CircleUnique>();
  node->input(input_nodes[0]);

  const auto *options = op.builtin_options.AsUniqueOptions();
  node->output_type(luci_datatype(options->idx_out_type));

  assert(int32_t(outputs.size()) == 2);
  // Let's use name of output 0 as Unique name
  const circle::TensorT &output_tensor = *tensors[outputs[0]];
  node->name(tensor_name(output_tensor));

  // Create virtual outputs of Unique
  for (int32_t n = 0; n < 2; ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleUniqueOut>();
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

CircleNode *CircleUniqueGraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  auto node = bna.context->graph()->nodes()->create<CircleUnique>();

  node->input(bna.input_nodes[0]);

  const auto *options = bna.op.builtin_options.AsUniqueOptions();
  node->output_type(luci_datatype(options->idx_out_type));

  return node;
}

CircleNode *CircleUniqueGraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleUniqueOut>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
