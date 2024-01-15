/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ShapeInfer_StridedSlice.h"
#include "Check.h"
#include "CircleShapeInferenceHelper.h"

#include <luci/IR/CircleNode.h>
#include <loco/IR/DataType.h>
#include <loco/IR/NodeShape.h>
#include <oops/InternalExn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

// code referenced from
// https://github.com/tensorflow/tensorflow/blob/3f878cff5b698b82eea85db2b60d65a2e320850e/
//    tensorflow/lite/kernels/strided_slice.cc
//    tensorflow/lite/kernels/internal/strided_slice_logic.h

namespace
{

// This Op only supports 1-5D cases and since we use the reference 4D
// implementation, the 1-3D tensors are mapped to 4D.
const int kMaxDim = 5;

const loco::DataType S32 = loco::DataType::S32;

struct StridedSliceParams
{
  int8_t start_indices_count = 0;
  int32_t start_indices[kMaxDim];
  int8_t stop_indices_count = 0;
  int32_t stop_indices[kMaxDim];
  int8_t strides_count = 0;
  int32_t strides[kMaxDim];

  int16_t begin_mask = 0;
  int16_t ellipsis_mask = 0;
  int16_t end_mask = 0;
  int16_t new_axis_mask = 0;
  int16_t shrink_axis_mask = 0;
};

struct StridedSliceContext
{
  StridedSliceContext(const luci::CircleStridedSlice *node)
  {
    // check overflow issues
    assert(static_cast<int16_t>(node->begin_mask()) == node->begin_mask());
    assert(static_cast<int16_t>(node->ellipsis_mask()) == node->ellipsis_mask());
    assert(static_cast<int16_t>(node->end_mask()) == node->end_mask());
    assert(static_cast<int16_t>(node->new_axis_mask()) == node->new_axis_mask());
    assert(static_cast<int16_t>(node->shrink_axis_mask()) == node->shrink_axis_mask());

    params.begin_mask = node->begin_mask();
    params.ellipsis_mask = node->ellipsis_mask();
    params.end_mask = node->end_mask();
    params.new_axis_mask = node->new_axis_mask();
    params.shrink_axis_mask = node->shrink_axis_mask();

    input = loco::must_cast<luci::CircleNode *>(node->input());
    begin = loco::must_cast<luci::CircleConst *>(node->begin());
    end = loco::must_cast<luci::CircleConst *>(node->end());
    strides = loco::must_cast<luci::CircleConst *>(node->strides());

    loco::TensorShape input_shape = luci::shape_get(input).as<loco::TensorShape>();
    input_dims = input_shape.rank();
  }
  StridedSliceParams params;
  luci::CircleNode *input = nullptr;
  luci::CircleConst *begin = nullptr;
  luci::CircleConst *end = nullptr;
  luci::CircleConst *strides = nullptr;

  // Equivalent input shape after adding axis according to new_axis_mask.
  loco::TensorShape effective_input_shape;
  int64_t input_dims = 0;
};

// Use until std::clamp() is available from C++17.
inline int Clamp(const int32_t v, const int32_t lo, const int32_t hi)
{
  LUCI_ASSERT(!(hi < lo), "Clamp hi < lo");
  if (hi < v)
    return hi;
  if (v < lo)
    return lo;
  return v;
}

// Return the index for the first element along that axis. This index will be a
// positive integer between [0, axis_size - 1] that can be used to index
// directly into the data.
inline int64_t StartForAxis(const StridedSliceParams &params, const loco::TensorShape &input_shape,
                            int64_t axis)
{
  const auto begin_mask = params.begin_mask;
  const auto *start_indices = params.start_indices;
  const auto *strides = params.strides;
  const int64_t axis_size = static_cast<int64_t>(input_shape.dim(axis).value());
  if (axis_size == 0)
  {
    return 0;
  }
  // Begin with the specified index.
  int64_t start = start_indices[axis];

  // begin_mask override
  if (begin_mask & (1 << axis))
  {
    if (strides[axis] > 0)
    {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axis_size-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = std::numeric_limits<int32_t>::lowest();
    }
    else
    {
      // Backward iteration - use the last element.
      start = std::numeric_limits<int32_t>::max();
    }
  }

  // Handle negative indices
  if (start < 0)
  {
    start += axis_size;
  }

  // Clamping
  if (strides[axis] > 0)
  {
    // Forward iteration
    start = Clamp(start, 0, axis_size);
  }
  else
  {
    // Backward iteration
    start = Clamp(start, -1, axis_size - 1);
  }

  return start;
}

// Return the "real" index for the end of iteration along that axis. This is an
// "end" in the traditional C sense, in that it points to one past the last
// element. ie. So if you were iterating through all elements of a 1D array of
// size 4, this function would return 4 as the stop, because it is one past the
// "real" indices of 0, 1, 2 & 3.
inline int64_t StopForAxis(const StridedSliceParams &params, const loco::TensorShape &input_shape,
                           int64_t axis, int64_t start_for_axis)
{
  const auto end_mask = params.end_mask;
  const auto shrink_axis_mask = params.shrink_axis_mask;
  const auto *stop_indices = params.stop_indices;
  const auto *strides = params.strides;
  const int64_t axis_size = static_cast<int64_t>(input_shape.dim(axis).value());
  if (axis_size == 0)
  {
    return 0;
  }

  // Begin with the specified index
  const bool shrink_axis = shrink_axis_mask & (1 << axis);
  int64_t stop = stop_indices[axis];

  // When shrinking an axis, the end position does not matter (and can be
  // incorrect when negative indexing is used, see Issue #19260). Always use
  // start_for_axis + 1 to generate a length 1 slice, since start_for_axis has
  // already been adjusted for negative indices.
  if (shrink_axis)
  {
    return start_for_axis + 1;
  }

  // end_mask override
  if (end_mask & (1 << axis))
  {
    if (strides[axis] > 0)
    {
      // Forward iteration - use the last element. These values will get
      // clamped below
      stop = std::numeric_limits<int32_t>::max();
    }
    else
    {
      // Backward iteration - use the first element.
      stop = std::numeric_limits<int32_t>::lowest();
    }
  }

  // Handle negative indices
  if (stop < 0)
  {
    stop += axis_size;
  }

  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (strides[axis] > 0)
  {
    // Forward iteration
    stop = Clamp(stop, 0, axis_size);
  }
  else
  {
    // Backward iteration
    stop = Clamp(stop, -1, axis_size - 1);
  }

  return stop;
}

StridedSliceParams BuildStridedSliceParams(StridedSliceContext *op_context)
{
  StridedSliceParams op_params;

  // The ellipsis_mask and new_axis_mask in op_params are not used. Those masks
  // are processed here to update begin_mask, end_mask and the index range.
  op_params.begin_mask = 0;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = 0;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = 0;

  // Count indexes where the new_axis_mask is set but the ellipsis_mask is not.
  loco::TensorShape begin_shape = luci::shape_get(op_context->begin).as<loco::TensorShape>();
  const int64_t begin_count = static_cast<int64_t>(begin_shape.dim(0).value());
  int64_t num_add_axis = 0;
  for (int64_t i = 0; i < begin_count; ++i)
  {
    if (!((1 << i) & op_context->params.ellipsis_mask) &&
        ((1 << i) & op_context->params.new_axis_mask))
    {
      num_add_axis++;
    }
  }

  // Calculate the dims of input after adding new axises.
  const int64_t effective_dims = op_context->input_dims + num_add_axis;

  // If begin, end and strides are not fully provided, it means Ellipsis should
  // be expanded to multiple dimensions (Ex: for spec [Ellipsis, 2] on a 3D
  // input, the Ellipsis should be applied for the first 2 dimensions). Besides,
  // If the new_axis_mask and the ellipsis_mask are set at the same index, the
  // new_axis_mask will have no effect.
  int64_t effective_ellipsis_mask = 0, effective_new_axis_mask = 0;
  int64_t ellipsis_start_idx = effective_dims, expanded_ellipsis = 0;
  for (int64_t i = 0; i < effective_dims;)
  {
    if ((1 << i) & op_context->params.ellipsis_mask)
    {
      ellipsis_start_idx = i;
      int64_t ellipsis_end_idx =
        std::max(i + 1, std::min(i + 1 + num_add_axis + op_context->input_dims - begin_count,
                                 effective_dims));
      expanded_ellipsis = ellipsis_end_idx - ellipsis_start_idx - 1;

      // Set bit for effective_ellipsis_mask.
      for (; i < ellipsis_end_idx; ++i)
      {
        effective_ellipsis_mask |= (1 << i);
      }
      continue;
    }

    if ((1 << (i - expanded_ellipsis)) & op_context->params.new_axis_mask)
    {
      effective_new_axis_mask |= (1 << i);
    }
    ++i;
  }

  // Calculate effective_input_shape and its corresponding begin, end, strides.
  loco::TensorShape input_shape = luci::shape_get(op_context->input).as<loco::TensorShape>();
  int64_t added_ellipsis = 0, added_axises = 0;
  op_context->effective_input_shape.rank(effective_dims);

  for (int64_t i = 0; i < effective_dims; ++i)
  {
    if ((1 << i) & effective_ellipsis_mask)
    {
      // If ellipsis_mask, set the begin_mask and end_mask at that index.
      added_ellipsis = std::max(int64_t(0), i - ellipsis_start_idx);
      assert(i < 16);
      op_params.begin_mask |= (1 << i);
      op_params.end_mask |= (1 << i);
      op_params.strides[i] = 1;
      op_context->effective_input_shape.dim(i) = input_shape.dim(i - added_axises);
    }
    else if ((1 << i) & effective_new_axis_mask)
    {
      // If new_axis_mask is set, it is equivalent to adding a new dim of 1 to
      // input tensor. Store added shape to effective_input_shape.
      op_params.start_indices[i] = 0;
      op_params.stop_indices[i] = 1;
      op_params.strides[i] = 1;
      op_context->effective_input_shape.dim(i) = loco::Dimension(1);
      added_axises++;
    }
    else if (i >= begin_count + expanded_ellipsis)
    {
      op_params.start_indices[i] = 0;
      op_params.stop_indices[i] = 0;
      op_params.strides[i] = 1;
      assert(i < 16);
      op_params.begin_mask |= (1 << i);
      op_params.end_mask |= (1 << i);
      op_context->effective_input_shape.dim(i) = input_shape.dim(i - added_axises);
    }
    else
    {
      const int64_t orig_idx = i - added_ellipsis;
      op_params.start_indices[i] = op_context->begin->at<S32>(orig_idx);
      op_params.stop_indices[i] = op_context->end->at<S32>(orig_idx);
      op_params.strides[i] = op_context->strides->at<S32>(orig_idx);
      if (op_context->params.begin_mask & (1 << orig_idx))
      {
        assert(i < 16);
        op_params.begin_mask |= (1 << i);
      }
      if (op_context->params.end_mask & (1 << orig_idx))
      {
        assert(i < 16);
        op_params.end_mask |= (1 << i);
      }
      if (op_context->params.shrink_axis_mask & (1 << orig_idx))
      {
        assert(i < 16);
        op_params.shrink_axis_mask |= (1 << i);
      }
      op_context->effective_input_shape.dim(i) = input_shape.dim(i - added_axises);
    }
  }

  // make sure no overflow
  assert(static_cast<int8_t>(effective_dims) == static_cast<int32_t>(effective_dims));

  op_params.start_indices_count = effective_dims;
  op_params.stop_indices_count = effective_dims;
  op_params.strides_count = effective_dims;

  return op_params;
}

} // namespace

namespace luci
{

loco::TensorShape infer_output_shape(const CircleStridedSlice *node)
{
  loco::TensorShape output_shape;

  auto input_node = loco::must_cast<luci::CircleNode *>(node->input());

  auto begin_node = dynamic_cast<luci::CircleConst *>(node->begin());
  auto end_node = dynamic_cast<luci::CircleConst *>(node->end());
  auto strides_node = dynamic_cast<luci::CircleConst *>(node->strides());
  if (begin_node == nullptr || end_node == nullptr || strides_node == nullptr)
  {
    INTERNAL_EXN("StridedSlice begin/end/strides nodes are not Constant");
  }

  LUCI_ASSERT(begin_node->dtype() == S32, "Only support S32 for begin_node");
  LUCI_ASSERT(end_node->dtype() == S32, "Only support S32 for end_node");
  LUCI_ASSERT(strides_node->dtype() == S32, "Only support S32 for strides_node");

  LUCI_ASSERT(begin_node->rank() == 1, "Only support rank 1 for begin_node");
  LUCI_ASSERT(end_node->rank() == 1, "Only support rank 1 for end_node");
  LUCI_ASSERT(strides_node->rank() == 1, "Only support rank 1 for strides_node");

  loco::TensorShape input_shape = luci::shape_get(input_node).as<loco::TensorShape>();

  assert(begin_node->size<S32>() <= input_shape.rank());
  assert(end_node->size<S32>() <= input_shape.rank());
  assert(strides_node->size<S32>() <= input_shape.rank());

  StridedSliceContext op_context(node);
  auto op_params = BuildStridedSliceParams(&op_context);
  auto &effective_input_shape = op_context.effective_input_shape;
  std::vector<int64_t> output_shape_vector;

  for (int32_t idx = effective_input_shape.rank() - 1; idx >= 0; --idx)
  {
    int32_t stride = op_params.strides[idx];
    LUCI_ASSERT(stride != 0, "stride value has to be non-zero");

    int64_t begin = StartForAxis(op_params, effective_input_shape, idx);
    int64_t end = StopForAxis(op_params, effective_input_shape, idx, begin);

    // When shrinking an axis, the end position does not matter (and can be
    // incorrect when negative indexing is used, see Issue #19260). Always use
    // begin + 1 to generate a length 1 slice, since begin has
    // already been adjusted for negative indices by GetBeginValueAtIndex.
    const bool shrink_axis = op_params.shrink_axis_mask & (1 << idx);
    if (shrink_axis)
    {
      end = begin + 1;
    }

    // This is valid for both positive and negative strides
    int64_t dim_shape = std::ceil((end - begin) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis)
    {
      output_shape_vector.push_back(dim_shape);
    }
  }

  auto shape_size = output_shape_vector.size();
  output_shape.rank(shape_size);
  for (uint32_t idx = 0; idx < shape_size; ++idx)
  {
    int64_t dim = output_shape_vector.at(shape_size - 1u - idx);
    LUCI_ASSERT(0 <= dim && dim < 0xfffffffL, "Dimension size exceeds limit");
    // reverse copy
    output_shape.dim(idx) = static_cast<uint32_t>(dim);
  }

  return output_shape;
}

} // namespace luci
