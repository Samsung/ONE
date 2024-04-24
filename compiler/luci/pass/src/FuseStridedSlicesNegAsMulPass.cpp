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

#include "luci/Pass/FuseStridedSlicesNegAsMulPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

// Create mul const if possible or return nullptr
luci::CircleConst *create_mul_const(luci::CircleStridedSlice *strided_slice_with_neg,
                                    luci::CircleConcatenation *concat_node)
{
  luci::CircleConst *begin_node =
    dynamic_cast<luci::CircleConst *>(strided_slice_with_neg->begin());
  luci::CircleConst *end_node = dynamic_cast<luci::CircleConst *>(strided_slice_with_neg->end());
  luci::CircleConst *strides_node =
    dynamic_cast<luci::CircleConst *>(strided_slice_with_neg->strides());

  assert(begin_node->dtype() == loco::DataType::S32);
  assert(end_node->dtype() == loco::DataType::S32);
  assert(strides_node->dtype() == loco::DataType::S32);

  auto ss_const_size = begin_node->size<loco::DataType::S32>();
  assert(ss_const_size = end_node->size<loco::DataType::S32>());
  assert(ss_const_size = strides_node->size<loco::DataType::S32>());

  // Check rank
  if (ss_const_size != concat_node->rank())
    return nullptr;

  // Check that strided slice with neg operation use only last dim of the input node
  // and all dims except last is 1
  for (uint32_t i = 0; i < concat_node->rank() - 1; ++i)
  {
    if (begin_node->at<loco::DataType::S32>(i) != 0 or
        end_node->at<loco::DataType::S32>(i) != concat_node->dim(i).value() or
        concat_node->dim(i).value() != 1)
      return nullptr;
  }

  assert(strided_slice_with_neg->dtype() == loco::DataType::FLOAT32);
  assert(concat_node->dtype() == loco::DataType::FLOAT32);

  auto new_node = concat_node->graph()->nodes()->create<luci::CircleConst>();
  new_node->name(concat_node->name() + strided_slice_with_neg->name() + "_const");
  new_node->dtype(loco::DataType::FLOAT32);
  new_node->rank(concat_node->rank());
  auto size = 1;
  for (uint32_t i = 0; i < new_node->rank(); i++)
  {
    new_node->dim(i).set(concat_node->dim(i).value());
    size *= new_node->dim(i).value();
  }
  new_node->size<loco::DataType::FLOAT32>(size);
  new_node->shape_status(luci::ShapeStatus::VALID);

  // Set 1 value for every node
  for (uint32_t i = 0; i < size; ++i)
  {
    new_node->at<loco::DataType::FLOAT32>(i) = 1.f;
  }

  uint32_t begin_index = begin_node->at<loco::DataType::S32>(concat_node->rank() - 1);
  uint32_t end_index = end_node->at<loco::DataType::S32>(concat_node->rank() - 1);
  uint32_t strides_index = strides_node->at<loco::DataType::S32>(concat_node->rank() - 1);
  for (uint32_t i = begin_index; i < end_index; i += strides_index)
  {
    new_node->at<loco::DataType::FLOAT32>(i) = -1.f;
  }

  return new_node;
}

/**
 *  Fuse StridedSlices Neg pattern as Mul if possible
 *
 *  BEFORE
 *                            |
 *                       [CircleNode]
 *                      |            |
 *        [CircleStridedSlice]  [CircleStridedSlice]
 *                     |             |
 *                     |        [CircleNeg]
 *                     |            |
 *                  [CircleConcatenation]
 *                            |
 *
 *  AFTER
 *                            |
 *                       [CircleNode]
 *                            |
 *                       [CircleMul] ------- [CircleConst]
 *                            |
 *
 *  Note: After the transformation, the CircleConst consists of 1 and -1,
 *  so that -1 appears for the part of the input tensor that is the input
 *  to the CircleNeg operation after applying StridedSlice.
 *
 *  Note: At the moment, only the case is supported when the slice occurs only
 *  in the last dimension, and all the others are equal to one.
 *  TODO: support other cases
 *
 */
bool fuse_strided_slices_neg_as_mul(luci::CircleConcatenation *concat)
{
  if (concat->numValues() != 2)
    return false;

  luci::CircleNeg *neg = nullptr;
  luci::CircleStridedSlice *first_strided_slice = nullptr;

  for (uint32_t i = 0; i < concat->numValues(); ++i)
  {
    neg = dynamic_cast<luci::CircleNeg *>(concat->values(0));
    first_strided_slice = dynamic_cast<luci::CircleStridedSlice *>(concat->values(1));

    if (neg == nullptr and first_strided_slice == nullptr)
    {
      neg = dynamic_cast<luci::CircleNeg *>(concat->values(1));
      first_strided_slice = dynamic_cast<luci::CircleStridedSlice *>(concat->values(0));
    }
  }

  if (neg == nullptr or first_strided_slice == nullptr)
    return false;

  luci::CircleStridedSlice *second_strided_slice =
    dynamic_cast<luci::CircleStridedSlice *>(neg->x());

  if (second_strided_slice == nullptr)
    return false;

  // Check strided slices have common input node
  luci::CircleNode *first_strided_slice_input =
    dynamic_cast<luci::CircleNode *>(first_strided_slice->input());
  luci::CircleNode *second_strided_slice_input =
    dynamic_cast<luci::CircleNode *>(second_strided_slice->input());
  if (first_strided_slice == nullptr or second_strided_slice == nullptr or
      first_strided_slice_input != second_strided_slice_input)
    return false;

  // TODO: add more types
  if (first_strided_slice->dtype() != loco::DataType::FLOAT32 or
      second_strided_slice->dtype() != loco::DataType::FLOAT32)
    return false;

  // Check first strided slice's begin, end and strides are const
  luci::CircleConst *begin_first_ss_node =
    dynamic_cast<luci::CircleConst *>(first_strided_slice->begin());
  luci::CircleConst *end_first_ss_node =
    dynamic_cast<luci::CircleConst *>(first_strided_slice->end());
  luci::CircleConst *strides_first_ss_node =
    dynamic_cast<luci::CircleConst *>(first_strided_slice->strides());
  if (begin_first_ss_node == nullptr or end_first_ss_node == nullptr or
      strides_first_ss_node == nullptr)
    return false;

  // Check second strided slice's begin, end and strides are const
  luci::CircleConst *begin_second_ss_node =
    dynamic_cast<luci::CircleConst *>(second_strided_slice->begin());
  luci::CircleConst *end_second_ss_node =
    dynamic_cast<luci::CircleConst *>(second_strided_slice->end());
  luci::CircleConst *strides_second_ss_node =
    dynamic_cast<luci::CircleConst *>(second_strided_slice->strides());
  if (begin_second_ss_node == nullptr or end_second_ss_node == nullptr or
      strides_second_ss_node == nullptr)
    return false;

  auto new_const = create_mul_const(second_strided_slice, concat);
  if (new_const == nullptr)
    return false;

  auto mul = concat->graph()->nodes()->create<luci::CircleMul>();
  mul->x(second_strided_slice_input);
  mul->y(new_const);
  mul->fusedActivationFunction(luci::FusedActFunc::NONE);
  mul->name(second_strided_slice->name() + first_strided_slice->name() + concat->name() +
            neg->name());
  luci::add_origin(mul, luci::get_origin(neg));

  loco::replace(concat).with(mul);

  return true;
}

} // namespace

namespace luci
{

bool FuseStridedSlicesNegAsMulPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto concat = dynamic_cast<luci::CircleConcatenation *>(node);
    if (not concat)
      continue;

    if (fuse_strided_slices_neg_as_mul(concat))
      changed = true;
  }

  return changed;
}

} // namespace luci
