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

#include "SqrtCanonicalizer.h"

#include <moco/IR/TFDialect.h>

namespace
{

bool canonicalize_sqrt(loco::Graph *graph, moco::TFSqrt *node)
{
  /**
   * @note This will replace TFSqrt node with Canonical EltwiseSqrt
   *
   *       Before
   *                 A --- TFSqrt -- C
   *       After
   *                    +- TFSqrt --
   *                    |
   *                 A -+- EltwiseSqrt -- C
   *
   *       Where
   *                 A : features of TFSqrt
   *                 C : a node that uses TFSqrt as an input
   *                 TFSqrt is disconnected from C
   */

  auto sqrt_node = graph->nodes()->create<loco::EltwiseSqrt>();

  auto node_A = node->x();

  // update connections
  sqrt_node->input(node_A);

  // replace node
  replace(node).with(sqrt_node);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool SqrtCanonicalizer::transform(TFSqrt *node) const
{
  return canonicalize_sqrt(node->graph(), node);
}

} // namespace tf
} // namespace moco
