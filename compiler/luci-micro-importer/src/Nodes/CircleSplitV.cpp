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
