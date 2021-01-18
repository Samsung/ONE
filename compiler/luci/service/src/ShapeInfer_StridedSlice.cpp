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
#include <loco/Service/ShapeInference.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace
{

// This Op only supports 1-4D cases and since we use the reference 4D
// implementation, the 1-3D tensors are mapped to 4D.
const int kMaxDim = 4;

const loco::DataType S32 = loco::DataType::S32;

using int8 = int8_t;
using int16 = int16_t;

struct StridedSliceParams
{
  int8 start_indices_count;
  int16 start_indices[kMaxDim];
  int8 stop_indices_count;
  int16 stop_indices[kMaxDim];
  int8 strides_count;
  int16 strides[kMaxDim];

  int16 begin_mask;
  int16 ellipsis_mask;
  int16 end_mask;
  int16 new_axis_mask;
  int16 shrink_axis_mask;
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
inline int StartForAxis(const StridedSliceParams &params, const loco::TensorShape &input_shape,
                        uint32_t axis)
{
  const auto begin_mask = params.begin_mask;
  const auto *start_indices = params.start_indices;
  const auto *strides = params.strides;
  const int32_t axis_size = static_cast<int>(input_shape.dim(axis).value());
  if (axis_size == 0)
  {
    return 0;
  }
  // Begin with the specified index.
  int32_t start = start_indices[axis];

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
  start = Clamp(start, 0, axis_size - 1);

  return start;
}

// Return the "real" index for the end of iteration along that axis. This is an
// "end" in the traditional C sense, in that it points to one past the last
// element. ie. So if you were iterating through all elements of a 1D array of
// size 4, this function would return 4 as the stop, because it is one past the
// "real" indices of 0, 1, 2 & 3.
inline int StopForAxis(const StridedSliceParams &params, const loco::TensorShape &input_shape,
                       int axis, int start_for_axis)
{
  const auto end_mask = params.end_mask;
  const auto shrink_axis_mask = params.shrink_axis_mask;
  const auto *stop_indices = params.stop_indices;
  const auto *strides = params.strides;
  const int axis_size = static_cast<int32_t>(input_shape.dim(axis).value());
  if (axis_size == 0)
  {
    return 0;
  }

  // Begin with the specified index
  const bool shrink_axis = shrink_axis_mask & (1 << axis);
  int32_t stop = stop_indices[axis];

  // When shrinking an axis, the end position does not matter (and can be
  // incorrect when negative indexing is used, see Issue #19260). Always use
  // start_for_axis + 1 to generate a length 1 slice, since start_for_axis has
  // already been adjusted for negative indices.
  if (shrink_axis)
  {
    stop = start_for_axis + 1;
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

StridedSliceParams BuildStridedSliceParams(const luci::CircleStridedSlice *node)
{
  StridedSliceParams op_params;

  if (kMaxDim < node->rank())
  {
    INTERNAL_EXN_V("Cannot support StridedSlice rank > ", kMaxDim);
  }

  auto begin_node = loco::must_cast<luci::CircleConst *>(node->begin());
  auto end_node = loco::must_cast<luci::CircleConst *>(node->end());
  auto strides_node = loco::must_cast<luci::CircleConst *>(node->strides());

  uint32_t dims_count = begin_node->size<S32>();

  op_params.start_indices_count = dims_count;
  op_params.stop_indices_count = dims_count;
  op_params.strides_count = dims_count;

  for (uint32_t i = 0; i < dims_count; ++i)
  {
    op_params.start_indices[i] = begin_node->at<S32>(i);
    op_params.stop_indices[i] = end_node->at<S32>(i);
    op_params.strides[i] = strides_node->at<S32>(i);
  }

  op_params.begin_mask = node->begin_mask();
  op_params.ellipsis_mask = 0;
  op_params.end_mask = node->end_mask();
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = node->shrink_axis_mask();

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

  assert(node->ellipsis_mask() == 0);
  assert(node->new_axis_mask() == 0);

  auto op_params = BuildStridedSliceParams(node);
  loco::TensorShape input_shape = luci::shape_get(input_node).as<loco::TensorShape>();

  uint32_t num_input_axes = input_shape.rank();
  assert(begin_node->size<S32>() <= num_input_axes);
  assert(end_node->size<S32>() <= num_input_axes);
  assert(strides_node->size<S32>() <= num_input_axes);
  for (uint32_t i = 0; i < strides_node->size<S32>(); i++)
  {
    LUCI_ASSERT(strides_node->at<S32>(i) != 0, "Stride value has to be non-zero");
  }

  uint32_t shape_size = 0;
  std::array<int32_t, 16> output_shape_data;

  for (uint32_t idx = 0; idx < num_input_axes; ++idx)
  {
    int32_t begin = StartForAxis(op_params, input_shape, idx);
    int32_t end = StopForAxis(op_params, input_shape, idx, begin);
    if (end < 0)
      end = input_shape.dim(idx).value() + end + 1;

    // This is valid for both positive and negative strides
    int32_t stride = strides_node->at<S32>(idx);
    int32_t dim_shape = std::ceil(static_cast<float>(end - begin) / stride);
    assert(dim_shape > 0);

    // When shrinking an axis, the end position does not matter (and can be
    // incorrect when negative indexing is used, see Issue #19260). Always use
    // begin + 1 to generate a length 1 slice, since begin has
    // already been adjusted for negative indices by StartForAxis.
    const bool shrink_axis = node->shrink_axis_mask() & (1 << idx);
    if (shrink_axis)
    {
      assert(dim_shape == 1);
    }
    else
    {
      output_shape_data[shape_size++] = dim_shape;
    }
  }

  output_shape.rank(shape_size);
  for (uint32_t idx = 0; idx < shape_size; ++idx)
  {
    output_shape.dim(idx) = output_shape_data[idx];
  }

  return output_shape;
}

} // namespace luci
