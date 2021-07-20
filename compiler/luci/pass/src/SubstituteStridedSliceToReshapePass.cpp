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

#include "luci/Pass/SubstituteStridedSliceToReshapePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <bitset>
#include <vector>

/**
 * @brief Convert strided_slice op in a certain condition to reshape op
 * @details Convert strided_slice op if the op meets all of the following condition:
 *          For all i, 0 <= i < input.rank
 *            - begin[i] == 0
 *            - end[i] >= input.shape.dim[i]
 *            - strides[i] == 1
 *          For all k (0 <= k < input.rank) where kth bit of shrink_axis_mask == 1
 *            - end[k] == 1
 *
 *          Example:
 *             input.shape = [1,1,2,3]
 *             strided_slice(input, begin=[0,0,0,0], end=[1,1,2,3], strides=[1,1,1,1],
 *                           shrink_axis_mask=0011b) // k = 0, 1
 *
 *             can be converted to
 *
 *             reshape(input, [2,3])
 */
namespace
{

/**
 * @brief Return newly-created CircleConst whose rank is 1
 */
luci::CircleConst *build_rank1_const(loco::Graph *graph, const std::vector<uint32_t> &values)
{
  auto const_node = graph->nodes()->create<luci::CircleConst>();
  const_node->dtype(loco::DataType::S32);
  const_node->size<loco::DataType::S32>(values.size());
  const_node->shape_status(luci::ShapeStatus::VALID);
  const_node->rank(1);
  const_node->dim(0) = values.size();

  for (size_t i = 0; i < values.size(); i++)
  {
    const_node->at<loco::DataType::S32>(i) = values.at(i);
  }

  return const_node;
}

/**
 * @brief Return newly-created CircleReshape node
 */
luci::CircleNode *build_reshape(loco::Graph *graph, const std::string &name,
                                const std::shared_ptr<luci::CircleNodeOrigin> &origin,
                                luci::CircleNode *input, const std::vector<uint32_t> &new_shape)
{
  auto reshape_node = graph->nodes()->create<luci::CircleReshape>();
  reshape_node->tensor(input);
  reshape_node->name(name);
  luci::add_origin(reshape_node, origin);

  auto new_shape_const = build_rank1_const(graph, new_shape);
  {
    new_shape_const->name(name + "/new_shape");
    luci::add_origin(new_shape_const, origin);
  }

  reshape_node->shape(new_shape_const);

  return reshape_node;
}

/**
 * @brief Return value in position on CircleConst with int64 format.
 */
int64_t value_from_circle_const(const luci::CircleConst *node, uint32_t idx)
{
  assert(node->rank() == 1 && node->dim(0).value() > idx);
  assert(node->dtype() == loco::DataType::S64 || node->dtype() == loco::DataType::S32);

  if (node->dtype() == loco::DataType::S64)
    return node->at<loco::DataType::S64>(idx);
  return static_cast<int64_t>(node->at<loco::DataType::S32>(idx));
}

bool substitute_strided_slice_to_reshape(luci::CircleStridedSlice *ss_node)
{
  if (ss_node->shrink_axis_mask() == 0)
    return false;

  // TODO Consider cases with ellipsis_mask and new_axis_mask
  // NOT YET SUPPORTED
  if (ss_node->ellipsis_mask() != 0 or ss_node->new_axis_mask() != 0)
    return false;

  auto begin_const = dynamic_cast<luci::CircleConst *>(ss_node->begin());
  auto strides_const = dynamic_cast<luci::CircleConst *>(ss_node->strides());
  auto end_const = dynamic_cast<luci::CircleConst *>(ss_node->end());

  if (not(begin_const && strides_const && end_const))
    return false;

  auto input_node = loco::must_cast<luci::CircleNode *>(ss_node->input());

  // condition check
  std::bitset<32> begin_mask(ss_node->begin_mask());
  std::bitset<32> end_mask(ss_node->end_mask());
  std::bitset<32> shrink_axis_mask(ss_node->shrink_axis_mask());

  uint input_rank = input_node->rank();
  for (uint32_t i = 0; i < input_rank; i++)
  {
    if (!input_node->dim(i).known())
      return false;

    auto begin_dim = value_from_circle_const(begin_const, i);
    if (begin_dim != 0 and begin_mask.test(i) == false)
      return false;

    // NOTE:
    //    In Tensorflow and TFLite, e.g., if input_shape = [2,3],
    //    strided_slice.end = [10,20] (larger value than actual dim)
    //    is treated as strided_slice.end = [2,3]
    int64_t end_dim = value_from_circle_const(end_const, i);
    if (end_dim < input_node->dim(i).value() and end_mask.test(i) == false)
      return false;

    int64_t strides_value = value_from_circle_const(strides_const, i);
    if (strides_value != 1)
      return false;

    if (shrink_axis_mask.test(i) && input_node->dim(i).value() != 1)
      return false;
  }

  // build shape for Reshape op
  bool found = false;
  std::vector<uint32_t> shrunk_shape;
  for (uint32_t i = 0; i < input_rank; i++)
  {
    if (input_node->dim(i) == 1 and shrink_axis_mask.test(i))
      found = true;
    else
      shrunk_shape.emplace_back(input_node->dim(i).value());
  }

  if (not found)
    return false;

  auto reshape_node = build_reshape(input_node->graph(), ss_node->name(), luci::get_origin(ss_node),
                                    input_node, shrunk_shape);

  replace(ss_node).with(reshape_node);
  return true;
}

} // namespace

namespace luci
{

/**
 * BEFORE
 *          |
 *     [CircleNode]  [CircleConst] [CircleConst] [CircleConst]
 *          |             |              |             |
 *          -------+------------------------------------
 *                 |
 *          [CircleStridedSlice]
 *                 |
 *            [CircleNode]
 *                 |
 * AFTER
 *                          |
 *     [CircleConst]  [CircleNode]    [CircleConst] [CircleConst] [CircleConst]
 *           \             /   \            |             |              |
 *          [CircleReshape]     -------------------+----------------------
 *                 |                               |
 *            [CircleNode]                [CircleStridedSlice]
 *                 |
 */
bool SubstituteStridedSliceToReshapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto circle_node = dynamic_cast<luci::CircleStridedSlice *>(node))
    {
      if (substitute_strided_slice_to_reshape(circle_node))
      {
        changed = true;
      }
    }
  }
  return changed;
}

} // namespace luci
