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

#include "luci/Import/Nodes/CircleBCQFullyConnected.h"

#include <luci/IR/Nodes/CircleBCQFullyConnected.h>

#include <loco.h>

namespace luci
{

bool CircleBCQFullyConnectedGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 5)
    return false;

  return true;
}

CircleNode *CircleBCQFullyConnectedGraphBuilder::build_node(const circle::OperatorT &op,
                                                            const std::vector<CircleNode *> &inputs,
                                                            loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleBCQFullyConnected>();

  node->input(inputs.at(0));
  node->weights_scales(inputs.at(1));
  node->weights_binary(inputs.at(2));
  node->bias(inputs.at(3));
  node->weights_clusters(inputs.at(4));

  // TODO Find and move to appropriate place for setting optional input
  if (auto bias = dynamic_cast<luci::CircleOutputExclude *>(node->bias()))
  {
    // bias is not used for type inference, but node itself should have a type
    bias->dtype(loco::DataType::FLOAT32);

    // bias is not used for shape inference
  }

  const auto *options = op.builtin_options.AsBCQFullyConnectedOptions();
  node->weights_hidden_size(options->weights_hidden_size);
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));

  return node;
}

} // namespace luci
