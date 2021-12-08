/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/CircleQuantParam.h"
#include "luci/Service/CircleNodeClone.h"

#include "CircleCloneNode.h"

#include <oops/UserExn.h>

#include <cassert>

namespace luci
{

/**
 * @note  Attributes of specific node type like keep_dims() of CircleSum are
 *        not copied.
 */
void copy_common_attributes(const luci::CircleNode *src, luci::CircleNode *dst)
{
  assert(src != nullptr);
  assert(dst != nullptr);

  dst->name(src->name());
  dst->dtype(src->dtype());

  dst->rank(src->rank());
  for (uint32_t i = 0; i < src->rank(); i++)
  {
    dst->dim(i) = src->dim(i);
  }
  dst->shape_status(src->shape_status());

  // quantparam
  copy_quantparam(src, dst);

  // sparsity
  const auto *sparsity = src->sparsityparam();
  if (sparsity != nullptr)
  {
    auto sparam = std::make_unique<luci::SparsityParam>();
    sparam->traversal_order = sparsity->traversal_order;
    sparam->block_map = sparsity->block_map;
    sparam->dim_metadata = sparsity->dim_metadata;

    dst->sparsityparam(std::move(sparam));
  }

  // op version
  dst->op_version(src->op_version());
}

/**
 * @note  Each visit implementation must copy node specific attributes.
 */
luci::CircleNode *clone_node(const luci::CircleNode *node, loco::Graph *graph)
{
  if (node == nullptr || graph == nullptr)
    return nullptr;

  CloneNode cn(graph);
  auto cloned = node->accept(&cn);
  if (cloned != nullptr)
    copy_common_attributes(node, cloned);
  return cloned;
}

} // namespace luci
