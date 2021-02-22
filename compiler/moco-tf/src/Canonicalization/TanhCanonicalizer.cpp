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

#include "TanhCanonicalizer.h"

#include <moco/IR/TFDialect.h>

namespace
{

bool canonicalize_tanh(loco::Graph *graph, moco::TFTanh *node)
{
  /**
   * @note This will replace TFTanh node with Canonical Tanh
   *
   *       Before
   *                 A --- TFTanh -- C
   *       After
   *                    +- TFTanh --
   *                    |
   *                 A -+-- Tanh --- C
   *
   *       Where
   *                 A : x of TFTanh
   *                 C : a node that uses TFTanh as an input
   *                 TFTanh is disconnected from C
   */

  auto tanh_node = graph->nodes()->create<loco::Tanh>();

  auto node_A = node->x();

  // update connections
  tanh_node->input(node_A);

  // replace node
  replace(node).with(tanh_node);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool TanhCanonicalizer::transform(TFTanh *node) const
{
  return canonicalize_tanh(node->graph(), node);
}

} // namespace tf
} // namespace moco
