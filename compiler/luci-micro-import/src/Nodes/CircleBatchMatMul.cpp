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

#include "luci/Import/Nodes/CircleBatchMatMul.h"

#include <luci/IR/Nodes/CircleBatchMatMul.h>

namespace luci
{

bool CircleBatchMatMulGraphBuilder::validate(const ValidateArgs &args) const
{
  return GraphBuilder::validate(args, 2);
}

CircleNode *CircleBatchMatMulGraphBuilder::build_node(const circle::OperatorT &op,
                                                      const std::vector<CircleNode *> &inputs,
                                                      loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleBatchMatMul>();
  node->x(inputs.at(0));
  node->y(inputs.at(1));

  const auto *options = op.builtin_options.AsBatchMatMulOptions();
  node->adj_x(options->adjoint_lhs);
  node->adj_y(options->adjoint_rhs);

  return node;
}

} // namespace luci
