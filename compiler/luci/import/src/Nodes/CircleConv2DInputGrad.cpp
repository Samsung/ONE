/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleConv2DInputGrad.h"

#include <luci/IR/Nodes/CircleConv2DInputGrad.h>

#include <loco.h>

#include <cassert>

namespace luci
{

bool CircleConv2DInputGradGraphBuilder::validate(const ValidateArgs &args) const
{
  return GraphBuilder::validate(args, 2);
}

CircleNode *CircleConv2DInputGradGraphBuilder::build_node(const circle::OperatorT &op,
                                                 const std::vector<CircleNode *> &inputs,
                                                 loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleConv2DInputGrad>();
  node->input_grad(inputs.at(0));
  node->weight(inputs.at(1));

  const auto *options = op.builtin_options.AsConv2DInputGradOptions();
  node->padding(luci_padding(options->padding));
  node->stride()->w(options->stride_w);
  node->stride()->h(options->stride_h);
  node->dilation()->w(options->dilation_w_factor);
  node->dilation()->h(options->dilation_h_factor);

  return node;
}

} // namespace luci
