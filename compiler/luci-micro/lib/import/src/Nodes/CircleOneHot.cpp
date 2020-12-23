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

#include "luci/Import/Nodes/CircleOneHot.h"

#include <luci/IR/Nodes/CircleOneHot.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleOneHotGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  const auto *options = args.op.builtin_options.AsOneHotOptions();

  // Only 4 Input come refered from
  if (inputs.size() != 4)
    return false;

  if (outputs.size() != 1)
    return false;

  const auto &tensors = args.reader.tensors();
  const auto &indices = tensors.at(inputs[0]);
  const auto &depth = tensors.at(inputs[1]);
  const auto &on_value = tensors.at(inputs[2]);
  const auto &off_value = tensors.at(inputs[3]);

  if (options->axis < -1 || options->axis > static_cast<int32_t>(indices->shape.size()))
    return false;
  if (depth->shape.size() != 0)
    return false;
  if (on_value->shape.size() != 0)
    return false;
  if (off_value->shape.size() != 0)
    return false;
  if (on_value->type != off_value->type)
    return false;

  return true;
}

CircleNode *CircleOneHotGraphBuilder::build_node(const circle::OperatorT &op,
                                                 const std::vector<CircleNode *> &inputs,
                                                 loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleOneHot>();

  node->indices(inputs[0]);
  node->depth(inputs[1]);
  node->on_value(inputs[2]);
  node->off_value(inputs[3]);

  const auto *options = op.builtin_options.AsOneHotOptions();
  node->axis(options->axis);

  return node;
}

} // namespace luci
