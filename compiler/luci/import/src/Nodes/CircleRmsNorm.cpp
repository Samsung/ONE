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

#include "luci/Import/Nodes/CircleRmsNorm.h"

#include <luci/IR/Nodes/CircleRmsNorm.h>

#include <loco.h>

namespace luci
{

bool CircleRmsNormGraphBuilder::validate(const ValidateArgs &args) const
{
  // TODO check dtypes
  return GraphBuilder::validate(args, 3);
}

CircleNode *CircleRmsNormGraphBuilder::build_node(const circle::OperatorT &op,
                                                  const std::vector<CircleNode *> &inputs,
                                                  loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleRmsNorm>();
  node->input(inputs.at(0));
  node->gamma(inputs.at(1));
  node->beta(inputs.at(2));

  const auto *options = op.builtin_options.AsRmsNormOptions();
  node->epsilon(options->epsilon);

  return node;
}

} // namespace luci
