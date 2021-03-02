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

#include "luci/Import/Nodes/CircleSplitV.h"

#include <luci/IR/Nodes/CircleSplitV.h>
#include <luci/IR/Nodes/CircleSplitVOut.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleSplitVGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  const auto *options = args.op.builtin_options.AsSplitVOptions();

  if (inputs.size() != 3)
    return false;

  if (static_cast<int32_t>(outputs.size()) != options->num_splits)
    return false;

  // TODO check types

  return true;
}

/**
 * @brief  SplitV Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple CircleSplitVOut nodes to emulate this
 *         For two outputs that may look like this
 *
 *         --- CircleSplitV --- FullyConnected ---
 *                           \- FullyConnected ---
 *
 *         will be created like this
 *
 *         --- CircleSplitV --- CircleSplitVOut --- FullyConnected ---
 *                           \- CircleSplitVOut --- FullyConnected ---
 */

#if 0
CircleNode *CircleSplitVGraphBuilder::build(const circle::OperatorT &op,
                                            GraphBuilderContext *context) const
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

  // Create CircleSplitV
  auto node = graph->nodes()->create<CircleSplitV>();
  node->input(input_nodes[0]);
  node->size_splits(input_nodes[1]);
  node->split_dim(input_nodes[2]);

  const auto *options = op.builtin_options.AsSplitVOptions();
  node->num_split(options->num_splits);

  assert(outputs.size() > 0);
  assert(int32_t(outputs.size()) == options->num_splits);
  {
    // Let's use name of output 0 as Split name
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    node->name(tensor_name(output_tensor));
    node->op_version(opcodes[op.opcode_index].get()->version);

    // NOTE We don't set quantization for Split itself but to virtual outputs
  }

  // Create virtual outputs of Split
  for (int32_t n = 0; n < options->num_splits; ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleSplitVOut>();
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

CircleNode *CircleSplitVGraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  auto node = bna.context->graph()->nodes()->create<CircleSplitV>();

  node->input(bna.input_nodes[0]);
  node->size_splits(bna.input_nodes[1]);
  node->split_dim(bna.input_nodes[2]);

  const auto *options = bna.op.builtin_options.AsSplitVOptions();
  node->num_split(options->num_splits);

  assert(int32_t(bna.op.outputs.size()) == options->num_splits);

  return node;
}

CircleNode *CircleSplitVGraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleSplitVOut>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
