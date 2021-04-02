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

#include "CircleNodeConnect.h"

#include <oops/UserExn.h>

namespace
{

// TODO remove CloneNode
class CloneNode final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNode(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleAdd *) final;
  luci::CircleNode *visit(const luci::CircleConst *) final;
  luci::CircleNode *visit(const luci::CircleDiv *) final;
  luci::CircleNode *visit(const luci::CircleMean *) final;
  luci::CircleNode *visit(const luci::CircleMul *) final;
  luci::CircleNode *visit(const luci::CirclePow *) final;
  luci::CircleNode *visit(const luci::CircleRsqrt *) final;
  luci::CircleNode *visit(const luci::CircleSqrt *) final;
  luci::CircleNode *visit(const luci::CircleSquaredDifference *) final;
  luci::CircleNode *visit(const luci::CircleSub *) final;
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

template <loco::DataType T>
void copy_values(const luci::CircleConst *node, luci::CircleConst *cloned)
{
  assert(T == node->dtype());
  assert(T == cloned->dtype());

  const auto size = node->size<T>();
  cloned->size<T>(size);
  for (uint32_t i = 0; i < size; i++)
    cloned->at<T>(i) = node->at<T>(i);
}

luci::CircleNode *CloneNode::visit(const luci::CircleConst *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleConst>();

  cloned->dtype(node->dtype());
  cloned->rank(node->rank());

  switch (node->dtype())
  {
    case loco::DataType::FLOAT32:
      copy_values<loco::DataType::FLOAT32>(node, cloned);
      break;

    case loco::DataType::U8:
      copy_values<loco::DataType::U8>(node, cloned);
      break;

    case loco::DataType::S8:
      copy_values<loco::DataType::S8>(node, cloned);
      break;

    case loco::DataType::S16:
      copy_values<loco::DataType::S16>(node, cloned);
      break;

    case loco::DataType::S32:
      copy_values<loco::DataType::S32>(node, cloned);
      break;

    case loco::DataType::S64:
      copy_values<loco::DataType::S64>(node, cloned);
      break;

    case loco::DataType::BOOL:
      copy_values<loco::DataType::BOOL>(node, cloned);
      break;

    default:
      throw oops::UserExn("Unsupported tensor dtype");
  }

  return cloned;
}

luci::CircleNode *CloneNode::visit(const luci::CircleDiv *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleDiv>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

luci::CircleNode *CloneNode::visit(const luci::CircleMean *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleMean>();
  cloned->keep_dims(node->keep_dims());
  return cloned;
}

luci::CircleNode *CloneNode::visit(const luci::CircleMul *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleMul>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

luci::CircleNode *CloneNode::visit(const luci::CirclePow *)
{
  return _graph->nodes()->create<luci::CirclePow>();
}

luci::CircleNode *CloneNode::visit(const luci::CircleRsqrt *)
{
  return _graph->nodes()->create<luci::CircleRsqrt>();
}

luci::CircleNode *CloneNode::visit(const luci::CircleSqrt *)
{
  return _graph->nodes()->create<luci::CircleSqrt>();
}

luci::CircleNode *CloneNode::visit(const luci::CircleSquaredDifference *)
{
  return _graph->nodes()->create<luci::CircleSquaredDifference>();
}

luci::CircleNode *CloneNode::visit(const luci::CircleSub *node)
{
  auto *cloned = _graph->nodes()->create<luci::CircleSub>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

} // namespace

namespace
{

class ConnectNode final : public luci::CircleNodeVisitor<void>
{
public:
  ConnectNode(luci::CloneContext &clonecontext) : _clonecontext(clonecontext){};

public:
  void visit(const luci::CircleAdd *) final;
  void visit(const luci::CircleConst *) final;
  void visit(const luci::CircleDiv *) final;
  void visit(const luci::CircleMean *) final;
  void visit(const luci::CircleMul *) final;
  void visit(const luci::CirclePow *) final;
  void visit(const luci::CircleRsqrt *) final;
  void visit(const luci::CircleSqrt *) final;
  void visit(const luci::CircleSquaredDifference *) final;
  void visit(const luci::CircleSub *) final;
  // TODO add all nodes

protected:
  luci::CircleNode *find_clone(const luci::CircleNode *node)
  {
    auto it = _clonecontext.find(node);
    if (it == _clonecontext.end())
      throw oops::UserExn("Invalid node in ConnectNode");
    return it->second;
  }

protected:
  luci::CloneContext &_clonecontext;
};

void ConnectNode::visit(const luci::CircleAdd *node)
{
  auto *cloned = loco::must_cast<luci::CircleAdd *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CircleConst *)
{
  // Nothing to do
}

void ConnectNode::visit(const luci::CircleDiv *node)
{
  auto *cloned = loco::must_cast<luci::CircleDiv *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CircleMean *node)
{
  auto *cloned = loco::must_cast<luci::CircleMean *>(find_clone(node));
  luci::CircleNode *in_i = loco::must_cast<luci::CircleNode *>(node->input());
  luci::CircleNode *in_r = loco::must_cast<luci::CircleNode *>(node->reduction_indices());
  cloned->input(find_clone(in_i));
  cloned->reduction_indices(find_clone(in_r));
}

void ConnectNode::visit(const luci::CircleMul *node)
{
  auto *cloned = loco::must_cast<luci::CircleMul *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CirclePow *node)
{
  auto *cloned = loco::must_cast<luci::CirclePow *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CircleRsqrt *node)
{
  auto *cloned = loco::must_cast<luci::CircleRsqrt *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  cloned->x(find_clone(in_x));
}

void ConnectNode::visit(const luci::CircleSqrt *node)
{
  auto *cloned = loco::must_cast<luci::CircleSqrt *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  cloned->x(find_clone(in_x));
}

void ConnectNode::visit(const luci::CircleSquaredDifference *node)
{
  auto *cloned = loco::must_cast<luci::CircleSquaredDifference *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CircleSub *node)
{
  auto *cloned = loco::must_cast<luci::CircleSub *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

} // namespace

namespace luci
{

// TODO remove this
void deprecated_copy_common_attributes(const luci::CircleNode *src, luci::CircleNode *dst)
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

// TODO remove this
luci::CircleNode *deprecated_clone_node(const luci::CircleNode *node, loco::Graph *graph)
{
  CloneNode cn(graph);
  auto cloned = node->accept(&cn);
  assert(cloned);
  deprecated_copy_common_attributes(node, cloned);
  return cloned;
}

void clone_connect(const CircleNode *node, CloneContext &clonecontext)
{
  ConnectNode cn(clonecontext);
  node->accept(&cn);
}

} // namespace luci
