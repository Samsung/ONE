/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "passes/optimizations/SinkRelu.h"
#include "passes/optimizations/OptimizationUtils.h"
#include "mir/ops/TransposeOp.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/Graph.h"
#include "mir/GraphPatternMatcher.h"

#include <algorithm>

namespace nnc
{

using namespace mir;
using namespace opt_util;

PassData SinkRelu::run(PassData data)
{
  auto g = static_cast<Graph *>(data);
  assert(g);
  GraphPatternMatcher matcher(g);
  auto is_relu = [](const Operation *op) { return op->getType() == Operation::Type::ReLU; };
  auto is_concat = [](const Operation *op) { return op->getType() == Operation::Type::concat; };
  auto is_max_pool = [](const Operation *op) {
    return op->getType() == Operation::Type::maxPool2D;
  };
  std::vector<std::pair<Operation *, Operation *>> matches;

  // sink ReLU through MaxPool
  matches = matcher.matchEdge(is_relu, is_max_pool);
  for (auto pair : matches)
  {
    swapAdjacent(g, pair.first, pair.second);
  }
  // sink ReLU through Concat
  auto matches_v = matcher.matchUpBush(is_relu, is_concat);
  for (const auto &pair : matches_v)
  {
    auto relus = pair.first;
    auto *concat = dynamic_cast<ops::ConcatOp *>(pair.second);
    std::vector<Operation::Output *> pre_relu;
    pre_relu.reserve(relus.size());
    for (auto *r : relus)
    {
      pre_relu.emplace_back(r->getInput(0));
    }
    // create replacement nodes
    auto new_concat = g->create<ops::ConcatOp>(pre_relu, concat->getAxis());
    auto new_relu = g->create<ops::ReluOp>(new_concat->getOutput(0));

    // concat is deleted here
    g->replaceNode(concat, new_relu);
    for (auto r : relus)
    {
      removeNodeIfUnused(g, r);
    }
  }
  return g;
}

} // namespace nnc
