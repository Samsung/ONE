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

#include "luci/Import/Nodes/CircleTransposeConv.h"

#include <luci/IR/Nodes/CircleTransposeConv.h>

#include <loco.h>

#include <cassert>

namespace luci
{

bool CircleTransposeConvGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 3)
    return false;

  return true;
}

CircleNode *CircleTransposeConvGraphBuilder::build_node(const circle::OperatorT &op,
                                                        const std::vector<CircleNode *> &inputs,
                                                        loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleTransposeConv>();

  node->inputSizes(inputs[0]);
  node->filter(inputs[1]);
  node->outBackprop(inputs[2]);

  const auto *options = op.builtin_options.AsTransposeConvOptions();
  node->padding(luci_padding(options->padding));
  node->stride()->w(options->stride_w);
  node->stride()->h(options->stride_h);

  return node;
}

} // namespace luci
