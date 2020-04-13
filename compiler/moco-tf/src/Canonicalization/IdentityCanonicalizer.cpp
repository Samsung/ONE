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

#include "IdentityCanonicalizer.h"

#include "Convert.h"

#include <moco/IR/TFDialect.h>

#include <moco/Names.h>
#include <moco/Log.h>

namespace
{

bool canonicalize_identity(loco::Graph *graph, moco::TFIdentity *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFIdentity node with Canonical Forward
   *
   *       Before
   *                 A -- TFIdentity -- C
   *
   *       After
   *                   /- TFIdentity --
   *                 A -- Forward -- C
   *
   *       Where
   *                 A : input of TFIdentity
   *                 C : a node that uses TFIdentity as an input
   *                 TFIdentity is disconnected from the output
   */

  INFO(l) << "TFNodeCanonicalize TFIdentity begin";

  auto forward_node = graph->nodes()->create<loco::Forward>();

  auto node_A = node->input();

  forward_node->input(node_A);

  // update graph
  replace(node).with(forward_node);

  INFO(l) << "TFNodeCanonicalize TFIdentity done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool IdentityCanonicalizer::transform(TFIdentity *node) const
{
  return canonicalize_identity(node->graph(), node);
}

} // namespace tf
} // namespace moco
