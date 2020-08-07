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

#include "luci/Import/Nodes/CircleAveragePool2D.h"

#include <luci/IR/Nodes/CircleAveragePool2D.h>

namespace luci
{

bool CircleAveragePool2DGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 1)
    return false;

  return true;
}

CircleNode *CircleAveragePool2DGraphBuilder::build_node(const circle::OperatorT &op,
                                                        const std::vector<CircleNode *> &inputs,
                                                        loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleAveragePool2D>();
  node->value(inputs.at(0));

  const auto *options = op.builtin_options.AsPool2DOptions();
  node->padding(luci_padding(options->padding));
  node->stride()->w(options->stride_w);
  node->stride()->h(options->stride_h);
  node->filter()->w(options->filter_width);
  node->filter()->h(options->filter_height);
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));

  return node;
}

} // namespace luci
