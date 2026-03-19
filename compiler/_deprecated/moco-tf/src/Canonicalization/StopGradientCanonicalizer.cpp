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

#include "StopGradientCanonicalizer.h"

#include <moco/IR/TFDialect.h>

#include <moco/Log.h>

namespace
{

bool canonicalize_stopgradient(loco::Graph *graph, moco::TFStopGradient *node)
{
  LOGGER(l);

  INFO(l) << "TFNodeCanonicalize TFStopGradient begin";

  /**
   * This will replace shape inferred TFStopGradient node into canonical Forward
   *
   * Before
   *           In --- TFStopGradient --- Out(s)
   *
   * After
   *               -- TFStopGradient
   *              /
   *           In --- Forward --- Out(s)
   */

  // Create loco node to replace
  auto forward_node = graph->nodes()->create<loco::Forward>();

  // update connection
  forward_node->input(node->input());

  // replace node
  replace(node).with(forward_node);

  INFO(l) << "TFNodeCanonicalize TFStopGradient done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool StopGradientCanonicalizer::transform(TFStopGradient *node) const
{
  return canonicalize_stopgradient(node->graph(), node);
}

} // namespace tf
} // namespace moco
