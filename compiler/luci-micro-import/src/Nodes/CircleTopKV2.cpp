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

#include "luci/Import/Nodes/CircleTopKV2.h"

#include <luci/IR/Nodes/CircleTopKV2.h>
#include <luci/IR/Nodes/CircleTopKV2Out.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleTopKV2GraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 2)
    return false;
  if (outputs.size() != 2)
    return false;

  const auto &tensors = args.reader.tensors();
  const auto &tensor = tensors.at(inputs.at(1));
  if (tensor->type != circle::TensorType_INT32)
    return false;

  return true;
}

/**
 * @brief  TopKV2 Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple CircleTopKV2Out nodes to emulate this
 *         For two outputs that may look like this
 *
 *         --- CircleTopKV2--- FullyConnected ---
 *                           \- FullyConnected ---
 *
 *         will be created like this
 *
 *         --- CircleTopKV2 --- CircleTopKV2Out --- FullyConnected ---
 *                           \- CircleTopKV2Out --- FullyConnected ---
 */

CircleNode *CircleTopKV2GraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  auto node = bna.context->graph()->nodes()->create<CircleTopKV2>();

  node->input(bna.input_nodes[0]);
  node->k(bna.input_nodes[1]);

  return node;
}

CircleNode *CircleTopKV2GraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleTopKV2Out>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
