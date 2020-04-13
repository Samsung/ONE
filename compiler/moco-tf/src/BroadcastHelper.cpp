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

#include "BroadcastHelper.h"

#include <loco/IR/Nodes.h>
#include <loco/Service/ShapeInference.h>

#include <cassert>

namespace
{

class NodeWithTensorShape
{
public:
  NodeWithTensorShape() = default;

public:
  NodeWithTensorShape(loco::Node *node, const loco::TensorShape &shape) : _node{node}, _shape{shape}
  {
    // DO NOTHING
  }

public:
  loco::Node *node(void) const { return _node; }
  const loco::TensorShape &shape(void) const { return _shape; }

private:
  loco::Node *_node = nullptr;
  loco::TensorShape _shape;
};

NodeWithTensorShape glue(loco::Node *node, const loco::TensorShape &shape)
{
  return NodeWithTensorShape(node, shape);
}

/**
 * @brief Create a higher-rank TensorShape following NumPy broadcasting semantics
 *
 * HOW TO USE:
 *
 *   auto expanded_tensor_shape = expand(tensor_shape).to(N);
 */
class TensorShapeExpander
{
public:
  TensorShapeExpander(const loco::TensorShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  loco::TensorShape to(uint32_t output_rank)
  {
    auto const &input_shape = _shape;
    uint32_t const input_rank = input_shape.rank();

    assert(input_rank <= output_rank && "Cannot shrink rank");
    uint32_t const axis_shift = output_rank - input_rank;

    loco::TensorShape output_shape;

    output_shape.rank(output_rank);
    for (uint32_t axis = 0; axis < output_rank; ++axis)
    {
      output_shape.dim(axis) = (axis < axis_shift) ? 1 : input_shape.dim(axis - axis_shift);
    }

    return output_shape;
  }

private:
  const loco::TensorShape _shape;
};

TensorShapeExpander expand(const loco::TensorShape &shape) { return TensorShapeExpander{shape}; }

/**
 * @brief Create a rank-expanded node (if required)
 */
class ExpandRankFunctor final
{
public:
  ExpandRankFunctor(uint32_t rank) : _rank{rank}
  {
    // DO NOTHING
  }

public:
  NodeWithTensorShape operator()(const NodeWithTensorShape &in) const
  {
    auto const input_node = in.node();
    auto const input_shape = in.shape();
    auto const input_rank = input_shape.rank();

    uint32_t const expected_rank = _rank;

    assert(input_rank <= expected_rank);
    if (input_rank == expected_rank)
    {
      // Nothing to expand
      return in;
    }

    auto g = input_node->graph();
    assert(g != nullptr);

    auto output_shape = expand(input_shape).to(expected_rank);
    auto output_node = g->nodes()->create<loco::FixedReshape>();

    output_node->input(input_node);
    output_node->rank(expected_rank);
    for (uint32_t axis = 0; axis < expected_rank; ++axis)
    {
      output_node->dim(axis) = output_shape.dim(axis);
    }

    return glue(output_node, output_shape);
  }

private:
  uint32_t _rank;
};

ExpandRankFunctor expand_rank_to(uint32_t rank) { return ExpandRankFunctor{rank}; }

/**
 * @brief Create a dimension-expanded node (if required)
 */
class ExpandDimsFunctor final
{
public:
  ExpandDimsFunctor(const loco::TensorShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  NodeWithTensorShape operator()(const NodeWithTensorShape &in) const
  {
    auto const input_node = in.node();
    auto const input_shape = in.shape();
    const auto &output_shape = _shape;

    assert(input_shape.rank() == output_shape.rank());

    if (input_shape == output_shape)
    {
      // Nothing to expand
      return in;
    }

    uint32_t const rank = output_shape.rank();

    auto g = input_node->graph();
    assert(g != nullptr);

    auto output_node = g->nodes()->create<loco::TensorBroadcast>();

    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      auto input_dim = input_shape.dim(axis);
      auto output_dim = output_shape.dim(axis);

      assert(input_dim.known() and output_dim.known());

      if (!(input_dim == output_dim))
      {
        assert(input_dim == 1);
        output_node->mapping()->dim(axis) = output_dim;
      }
    }

    output_node->input(input_node);

    return glue(output_node, output_shape);
  }

private:
  loco::TensorShape _shape;
};

ExpandDimsFunctor expand_dims_as(const loco::TensorShape &shape)
{
  return ExpandDimsFunctor{shape};
}

} // namespace

namespace moco
{
namespace tf
{

loco::Node *BroadcastFunctor::build(loco::Node *node, const loco::TensorShape &shape) const
{
  // clang-format off
  return glue(node, shape)
       | expand_rank_to(_shape.rank())
       | expand_dims_as(_shape)
       | [] (const NodeWithTensorShape &in) { return in.node(); };
  // clang-format on
}

loco::Node *BroadcastFunctor::build(loco::Node *node) const
{
  return build(node, loco::shape_get(node).as<loco::TensorShape>());
}

} // namespace tf
} // namespace moco
