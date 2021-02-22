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

#include "ReluCanonicalizer.h"

#include <moco/IR/TFDialect.h>

namespace
{

bool canonicalize_relu(loco::Graph *graph, moco::TFRelu *node)
{
  /**
   * @note This will replace TFRelu node with Canonical ReLU
   *
   *       Before
   *                 A --- TFRelu -- C
   *       After
   *                    +- TFRelu --
   *                    |
   *                 A -+- ReLU -- C
   *
   *       Where
   *                 A : features of TFRelu
   *                 C : a node that uses TFRelu as an input
   *                 TFRelu is disconnected from C
   */

  auto relu_node = graph->nodes()->create<loco::ReLU>();

  auto node_A = node->features();

  // update connections
  relu_node->input(node_A);

  // replace node
  replace(node).with(relu_node);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool ReluCanonicalizer::transform(TFRelu *node) const
{
  return canonicalize_relu(node->graph(), node);
}

} // namespace tf
} // namespace moco
