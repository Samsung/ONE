/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FoldStridedSlicePass.h"

#include "helpers/Compute.h"
#include "helpers/Shape.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <luci/Log.h>

#include <luci_compute/StridedSlice.h>

namespace luci
{

namespace
{

template <loco::DataType OutType>
bool set_params(const luci::CircleStridedSlice *node,
                compute::StridedSlice<typename loco::DataTypeImpl<OutType>::Type> &css,
                luci::CircleConst *begin_const, luci::CircleConst *end_const,
                luci::CircleConst *strides_const)
{
  assert(node);

  auto &params = css.params();

  // SET PARAMETERS
  params.start_indices_count = begin_const->size<OutType>();
  for (uint32_t i = 0; i < begin_const->size<OutType>(); ++i)
    params.start_indices[i] = begin_const->at<OutType>(i);
  params.stop_indices_count = end_const->size<OutType>();
  for (uint32_t i = 0; i < end_const->size<OutType>(); ++i)
    params.stop_indices[i] = end_const->at<OutType>(i);
  params.strides_count = strides_const->size<OutType>();
  for (uint32_t i = 0; i < strides_const->size<OutType>(); ++i)
    params.strides[i] = strides_const->at<OutType>(i);

  params.begin_mask = node->begin_mask();
  params.ellipsis_mask = node->ellipsis_mask();
  params.end_mask = node->end_mask();
  params.new_axis_mask = node->new_axis_mask();
  params.shrink_axis_mask = node->shrink_axis_mask();

  return true;
}

/**
 * Fold StridedSlice with constant input into a constant tensor
 *
 * BEFORE
 *
 *        [CircleConst]
 *              |
 *     [CircleStridedSlice]
 *              |
 *        [CircleNode]
 *
 * AFTER
 *
 *     [CircleConst]  [CircleConst]
 *           |
 *     [CircleNode]
 *
 */
template <loco::DataType OutType> bool fold_strided_slice(luci::CircleStridedSlice *strided_slice)
{
  auto input_node = dynamic_cast<luci::CircleConst *>(strided_slice->input());
  if (input_node == nullptr)
    return false; // Constant input is required for folding
  auto name = input_node->name();
  assert(name.length() > 0);

  auto begin_const = dynamic_cast<luci::CircleConst *>(strided_slice->begin());
  if (begin_const == nullptr)
    return false;
  auto end_const = dynamic_cast<luci::CircleConst *>(strided_slice->end());
  if (end_const == nullptr)
    return false;
  auto strides_const = dynamic_cast<luci::CircleConst *>(strided_slice->strides());
  if (strides_const == nullptr)
    return false;

  auto static_shape = [](luci::CircleNode *node) {
    loco::TensorShape shape;
    shape.rank(node->rank());
    for (uint32_t i = 0; i < node->rank(); ++i)
      shape.dim(i) = node->dim(i);
    return shape;
  };

  using PRIMITIVE_DTYPE = typename loco::DataTypeImpl<OutType>::Type;
  compute::StridedSlice<PRIMITIVE_DTYPE> comp_strided_slice{};
  if (!set_params<OutType>(strided_slice, comp_strided_slice, begin_const, end_const,
                           strides_const))
    return false;

  auto const input_data = &input_node->at<OutType>(0);
  auto const begin_data = &begin_const->at<OutType>(0);
  auto const end_data = &end_const->at<OutType>(0);
  auto const strides_data = &strides_const->at<OutType>(0);
  comp_strided_slice.input(static_shape(input_node), input_data);
  comp_strided_slice.begin(static_shape(begin_const), begin_data);
  comp_strided_slice.end(static_shape(end_const), end_data);
  comp_strided_slice.strides(static_shape(strides_const), strides_data);

  if (!comp_strided_slice.prepare())
    return false;

  auto output_shape = comp_strided_slice.output_shape();
  auto output_size = loco::element_count(&output_shape);

  // result folded constant node
  auto folded_strided_slice = input_node->graph()->nodes()->create<luci::CircleConst>();
  folded_strided_slice->name(name + "_ConstStridedSlice");
  folded_strided_slice->dtype(input_node->dtype());
  folded_strided_slice->rank(input_node->rank());
  folded_strided_slice->shape_status(luci::ShapeStatus::VALID);
  folded_strided_slice->size<OutType>(output_size);

  auto folded_data = &folded_strided_slice->at<OutType>(0);
  comp_strided_slice.output(folded_data);
  comp_strided_slice.compute();

  loco::replace(strided_slice).with(folded_strided_slice);

  return true;
}

} // namespace

/**
 * Constant Folding for StridedSlice Op
 **/
bool FoldStridedSlicePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto strided_slice = dynamic_cast<CircleStridedSlice *>(node))
    {
      auto out_type = strided_slice->dtype();
      switch (out_type)
      {
        // TODO support more data types
        case loco::DataType::S32:
          if (fold_strided_slice<loco::DataType::S32>(strided_slice))
            changed = true;
          break;
        case loco::DataType::FLOAT32:
          if (fold_strided_slice<loco::DataType::FLOAT32>(strided_slice))
            changed = true;
          break;
        default:
          break;
      }
    }
  }

  return changed;
}

} // namespace luci
