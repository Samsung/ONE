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

#include "CircleNodeClone.h"

#include <oops/UserExn.h>

namespace
{

class CloneNode final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNode(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleAdd *) final;
  // TODO add all nodes

  // NOTE CircleNodeVisitor will throw if not supported here

protected:
  loco::Graph *_graph = nullptr;
};

luci::CircleNode *CloneNode::visit(const luci::CircleAdd *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleAdd>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

} // namespace

namespace luci
{

void copy_common_attributes(const luci::CircleNode *src, luci::CircleNode *dst)
{
  assert(src != nullptr);
  assert(dst != nullptr);

  dst->name(src->name());
  dst->dtype(src->dtype());

  dst->rank(src->rank());
  for (uint32_t i = 0; i < src->rank(); i++)
  {
    if (src->dim(i).known())
      dst->dim(i).set(src->dim(i).value());
  }
  dst->shape_status(src->shape_status());

  // quantparam
  const auto *quantparam = src->quantparam();
  if (quantparam != nullptr)
  {
    auto qparam = std::make_unique<luci::CircleQuantParam>();
    qparam->scale = quantparam->scale;
    qparam->zerop = quantparam->zerop;
    qparam->min = quantparam->min;
    qparam->max = quantparam->max;
    qparam->quantized_dimension = quantparam->quantized_dimension;

    dst->quantparam(std::move(qparam));
  }

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

luci::CircleNode *clone_node(const luci::CircleNode *node, loco::Graph *graph)
{
  CloneNode cn(graph);
  auto cloned = node->accept(&cn);
  assert(cloned);
  copy_common_attributes(node, cloned);
  return cloned;
}

} // namespace luci
