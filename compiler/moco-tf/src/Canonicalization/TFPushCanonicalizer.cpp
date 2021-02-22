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

#include "TFPushCanonicalizer.h"

#include <moco/IR/TFDialect.h>

namespace
{

bool canonicalize_push(loco::Graph *graph, moco::TFPush *node)
{
  /**
   * @note This will replace TFRelu node with Canonical ReLU
   *
   *       Before
   *                 A --- TFPush
   *       After
   *                    +- TFPush
   *                    |
   *                 A -+- Push
   *
   *       Where
   *                 A : from of TFPush
   *                 TFPush will have no GraphOutputIndex
   *                 Push will have GraphOutputIndex that from TFPush
   */

  auto push_node = graph->nodes()->create<loco::Push>();

  auto node_A = node->from();

  // update connections
  push_node->from(node_A);

  // update output index
  push_node->index(node->index());
  node->index_reset();

  // replace node
  replace(node).with(push_node);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool TFPushCanonicalizer::transform(TFPush *node) const
{
  return canonicalize_push(node->graph(), node);
}

} // namespace tf
} // namespace moco
