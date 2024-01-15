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

#include "luci/Pass/SubstituteSqueezeToReshapePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

/**
 * @brief return TRUE if all dim is known
 * @note This pass can be applied even some of dimensions are unknown.
         For now, do not consider about it and update logic later.
 */
bool can_squeeze_shape(const luci::CircleNode *node)
{
  for (uint32_t r = 0; r < node->rank(); ++r)
  {
    if (not node->dim(r).known())
      return false;
  }
  return true;
}

/**
 * @brief return valid unsigned dim value from 0 ~ (rank-1)
 * @note  dim can be -rank to (rank-1)
 */
uint32_t valid_unsigned_dim(uint32_t rank, int32_t dim)
{
  int32_t irank = static_cast<int32_t>(rank);
  return dim >= 0 ? static_cast<uint32_t>(dim) : static_cast<uint32_t>(irank + dim);
}

/**
 * @brief return TRUE if input dim is 1 for squeeze_dims values
 */
bool is_valid_input(const luci::CircleNode *node, const std::vector<int32_t> &squeeze_dims)
{
  auto rank = node->rank();
  for (auto dim : squeeze_dims)
  {
    auto udim = valid_unsigned_dim(rank, dim);
    if (node->dim(udim).value() != 1)
      return false;
  }
  return true;
}

/**
 * @brief return shape vector from input
 */
std::vector<uint32_t> node_shape(const luci::CircleNode *input)
{
  std::vector<uint32_t> shape;
  uint32_t rank = input->rank();
  for (uint32_t r = 0; r < rank; ++r)
    shape.push_back(input->dim(r).value());

  return shape;
}

/**
 * @brief return CircleConst ptr with values of new_shape
 */
luci::CircleConst *create_shape_const(loco::Graph *graph, const std::vector<uint32_t> &new_shape)
{
  // NOTE dim_size can be 0
  uint32_t dim_size = static_cast<uint32_t>(new_shape.size());

  auto shape_const = graph->nodes()->create<luci::CircleConst>();

  // const shape/dtype
  shape_const->dtype(loco::DataType::S32);
  if (dim_size > 0)
  {
    shape_const->rank(1);
    shape_const->dim(0).set(dim_size);
  }
  else
    shape_const->rank(0);
  shape_const->shape_status(luci::ShapeStatus::VALID);

  // constant values
  shape_const->size<loco::DataType::S32>(dim_size);
  for (uint32_t i = 0; i < dim_size; ++i)
    shape_const->at<loco::DataType::S32>(i) = new_shape.at(i);

  return shape_const;
}

bool substitute_squeeze_to_reshape(luci::CircleSqueeze *squeeze)
{
  assert(squeeze != nullptr);

  auto input = loco::must_cast<luci::CircleNode *>(squeeze->input());
  // we need input node shape and all dim should be known
  if (input->shape_status() != luci::ShapeStatus::VALID)
    return false;
  if (not can_squeeze_shape(input))
    return false;

  // we will use squeeze shape for new shape
  if (squeeze->shape_status() != luci::ShapeStatus::VALID)
    return false;

  auto &squeeze_dims = squeeze->squeeze_dims();
  if (not is_valid_input(input, squeeze_dims))
    throw std::runtime_error("Invalid values in squeeze_dims: " + squeeze->name());

  auto name = squeeze->name();
  assert(name.length() > 0);

  auto reshape_shape = node_shape(squeeze);
  auto graph = squeeze->graph();
  auto reshape = graph->nodes()->create<luci::CircleReshape>();
  auto shape_const = create_shape_const(graph, reshape_shape);
  copy_quantparam(squeeze, reshape);
  reshape->name(name + "/Reshape");
  luci::add_origin(reshape, luci::get_origin(squeeze));
  shape_const->name(name + "/Reshape/shape");

  // graph connection
  reshape->tensor(input);
  reshape->shape(shape_const);
  replace(squeeze).with(reshape);

  return true;
}

} // namespace

namespace luci
{

/**
 * BEFORE
 *           |
 *      [CircleNode]
 *           |
 *    [CircleSqueeze]
 *           |
 *      [CircleNode]
 *           |
 *
 * AFTER
 *               |
 *          [CircleNode]  [CircleConst]
 *             |    \             /
 *  [CircleSqueeze] [CircleReshape]
 *                        |
 *                   [CircleNode]
 *                        |
 */
bool SubstituteSqueezeToReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto squeeze = dynamic_cast<luci::CircleSqueeze *>(node))
    {
      if (substitute_squeeze_to_reshape(squeeze))
        changed = true;
    }
  }
  return changed;
}

} // namespace luci
