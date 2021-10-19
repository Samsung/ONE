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

#include "luci/Import/Nodes/CircleSpaceToBatchND.h"

#include <luci/IR/Nodes/CircleSpaceToBatchND.h>

#include "ValidateHelpers.h"

#include <loco.h>

namespace luci
{

bool CircleSpaceToBatchNDGraphBuilder::validate(const ValidateArgs &args) const
{
  return validate_batch_space_nd(args);
}

CircleNode *CircleSpaceToBatchNDGraphBuilder::build_node(const circle::OperatorT &,
                                                         const std::vector<CircleNode *> &inputs,
                                                         loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleSpaceToBatchND>();
  node->input(inputs.at(0));
  node->block_shape(inputs.at(1));
  node->paddings(inputs.at(2));

  // No options for SpaceToBatchND

  return node;
}

} // namespace luci
