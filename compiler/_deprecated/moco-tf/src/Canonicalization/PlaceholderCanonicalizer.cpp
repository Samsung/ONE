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

#include "PlaceholderCanonicalizer.h"

#include <moco/IR/TFDialect.h>

#include <moco/Names.h>
#include <moco/Log.h>

namespace
{

bool canonicalize_placeholder(loco::Graph *graph, moco::TFPlaceholder *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFPlaceholder node with Canonical Pull
   *
   *       Before
   *                 TFPlaceholder -- C
   *
   *       After
   *                 TFPlaceholder -
   *                 Pull -- C
   *
   *       Where
   *                 C : a node that uses TFPlaceholder as an input
   *                 TFPlaceholder is disconnected from other nodes
   */

  INFO(l) << "PlaceholderCanonicalizer begin";

  auto pull_node = graph->nodes()->create<loco::Pull>();

  // copy properties
  auto dtype = node->dtype();
  pull_node->dtype(dtype);

  auto rank = node->rank();

  if (rank == 0)
  {
    // This routine implements a workaround that converts a scalar constant (rank-0 tensor)
    // into a rank-1 tensor of shape [1].
    //
    // TODO Revise this implementation later
    pull_node->rank(1);
    pull_node->dim(0) = 1;
  }
  else
  {
    pull_node->rank(rank);

    for (uint32_t r = 0; r < rank; ++r)
    {
      if (node->dim(r).known())
        pull_node->dim(r) = node->dim(r);
      else
        pull_node->dim(r).unset();
    }
  }

  // set loco::Pull GraphInputIndex
  pull_node->index(moco::index(node));

  // update graph
  replace(node).with(pull_node);

  INFO(l) << "PlaceholderCanonicalizer done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool PlaceholderCanonicalizer::transform(TFPlaceholder *node) const
{
  return canonicalize_placeholder(node->graph(), node);
}

} // namespace tf
} // namespace moco
