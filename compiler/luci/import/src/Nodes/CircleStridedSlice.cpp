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

#include "luci/Import/Nodes/CircleStridedSlice.h"

#include <luci/IR/Nodes/CircleStridedSlice.h>

#include <loco.h>

#include <cassert>

namespace luci
{

bool CircleStridedSliceGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 4)
    return false;
  if (args.op.outputs.size() != 1)
    return false;

  // TODO check shapes and types

  return true;
}

CircleNode *CircleStridedSliceGraphBuilder::build_node(const circle::OperatorT &op,
                                                       const std::vector<CircleNode *> &inputs,
                                                       loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleStridedSlice>();
  node->input(inputs.at(0));
  node->begin(inputs.at(1));
  node->end(inputs.at(2));
  node->strides(inputs.at(3));

  const auto *options = op.builtin_options.AsStridedSliceOptions();
  node->begin_mask(options->begin_mask);
  node->end_mask(options->end_mask);
  node->ellipsis_mask(options->ellipsis_mask);
  node->new_axis_mask(options->new_axis_mask);
  node->shrink_axis_mask(options->shrink_axis_mask);

  return node;
}

} // namespace luci
