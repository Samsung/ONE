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

#include "Relu6Canonicalizer.h"

#include <moco/IR/TFDialect.h>

namespace
{

bool canonicalize_relu6(loco::Graph *graph, moco::TFRelu6 *node)
{
  /**
   * @note This will replace TFRelu6 node with Canonical ReLU6
   *
   *       Before
   *                 A --- TFRelu6 -- C
   *       After
   *                    +- TFRelu6 --
   *                    |
   *                 A -+- ReLU6 -- C
   *
   *       Where
   *                 A : features of TFRelu6
   *                 C : a node that uses TFRelu6 as an input
   *                 TFRelu6 is disconnected from C
   */

  auto relu6_node = graph->nodes()->create<loco::ReLU6>();

  auto node_A = node->features();

  // update connections
  relu6_node->input(node_A);

  // replace node
  replace(node).with(relu6_node);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool Relu6Canonicalizer::transform(TFRelu6 *node) const
{
  return canonicalize_relu6(node->graph(), node);
}

} // namespace tf
} // namespace moco
