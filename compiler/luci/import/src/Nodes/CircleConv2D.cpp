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

#include "luci/Import/Nodes/CircleConv2D.h"

#include <luci/IR/Nodes/CircleConv2D.h>

#include <loco.h>

#include <cassert>

namespace luci
{

bool CircleConv2DGraphBuilder::validate(const ValidateArgs &args) const
{
  // Circle Conv2D may not have a bias but we won't support this
  if (args.op.inputs.size() != 3)
    return false;

  return true;
}

CircleNode *CircleConv2DGraphBuilder::build_node(const circle::OperatorT &op,
                                                 const std::vector<CircleNode *> &inputs,
                                                 loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleConv2D>();
  node->input(inputs.at(0));
  node->filter(inputs.at(1));
  // For now, bias is required (checked in `verify` method).
  assert(inputs.size() == 3);
  node->bias(inputs.at(2));

  const auto *options = op.builtin_options.AsConv2DOptions();
  node->padding(luci_padding(options->padding));
  node->stride()->w(options->stride_w);
  node->stride()->h(options->stride_h);
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));
  node->dilation()->w(options->dilation_w_factor);
  node->dilation()->h(options->dilation_h_factor);

  return node;
}

} // namespace luci
