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

#include "luci/Import/Nodes/CirclePadV2.h"

#include <luci/IR/Nodes/CirclePadV2.h>

#include <loco.h>

namespace luci
{

bool CirclePadV2GraphBuilder::validate(const ValidateArgs &args) const
{
  return GraphBuilder::validate(args, 3);
}

CircleNode *CirclePadV2GraphBuilder::build_node(const circle::OperatorT &op,
                                                const std::vector<CircleNode *> &inputs,
                                                loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CirclePadV2>();
  node->input(inputs[0]);
  node->paddings(inputs[1]);
  node->constant_values(inputs[2]);

  const auto *options = op.builtin_options.AsPadV2Options();
  (void)options; // There are no options.

  return node;
}

} // namespace luci
