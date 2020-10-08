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

#ifndef __NNFW_CKER_STRIDEDSLICE_H__
#define __NNFW_CKER_STRIDEDSLICE_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

#include <cmath>

namespace nnfw
{
namespace cker
{
// Use until std::clamp() is available from C++17.
inline int Clamp(const int v, const int lo, const int hi)
{
  assert(!(hi < lo));
  if (hi < v)
    return hi;
  if (v < lo)
    return lo;
  return v;
}

inline void StridedSlicePadIndices(StridedSliceParams *p, int dim_count)
{
  // Add indices and mask bits to fully include extra dimensions
  assert(dim_count <= 4);
  assert(dim_count >= p->start_indices_count);
  assert(p->start_indices_count == p->stop_indices_count);
  assert(p->stop_indices_count == p->strides_count);

  const int pad_count = dim_count - p->start_indices_count;

  // Pad indices at start, so move arrays by pad_count.
  for (int i = p->start_indices_count - 1; i >= 0; --i)
  {
    p->strides[i + pad_count] = p->strides[i];
    p->start_indices[i + pad_count] = p->start_indices[i];
    p->stop_indices[i + pad_count] = p->stop_indices[i];
  }
  for (int i = 0; i < pad_count; ++i)
  {
    p->start_indices[i] = 0;
    p->stop_indices[i] = 1;
    p->strides[i] = 1;
  }

  // Pad masks with 0s or 1s as required.
  p->shrink_axis_mask <<= pad_count;
  p->ellipsis_mask <<= pad_count;
  p->new_axis_mask <<= pad_count;
  p->begin_mask <<= pad_count;
  p->end_mask <<= pad_count;
  p->begin_mask |= (1 << pad_count) - 1;
  p->end_mask |= (1 << pad_count) - 1;

  p->start_indices_count = dim_count;
  p->stop_indices_count = dim_count;
  p->strides_count = dim_count;
}

// Return the index for the first element along that axis. This index will be a
// positive integer between [0, axis_size - 1] that can be used to index
// directly into the data.
inline int StartForAxis(const StridedSliceParams &params, const Shape &input_shape, int axis)
{
  const auto begin_mask = params.begin_mask;
  const auto *start_indices = params.start_indices;
  const auto *strides = params.strides;
  // Begin with the specified index.
  int start = start_indices[axis];

  // begin_mask override
  if (begin_mask & 1 << axis)
  {
    if (strides[axis] > 0)
    {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axis_size-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = std::numeric_limits<int>::lowest();
    }
    else
    {
      // Backward iteration - use the last element.
      start = std::numeric_limits<int>::max();
    }
  }

  // Handle negative indices
  int axis_size = input_shape.Dims(axis);
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
inline int StopForAxis(const StridedSliceParams &params, const Shape &input_shape, int axis,
                       int start_for_axis)
{
  const auto end_mask = params.end_mask;
  const auto shrink_axis_mask = params.shrink_axis_mask;
  const auto *stop_indices = params.stop_indices;
  const auto *strides = params.strides;

  // Begin with the specified index
  const bool shrink_axis = shrink_axis_mask & (1 << axis);
  int stop = stop_indices[axis];

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
      stop = std::numeric_limits<int>::max();
    }
    else
    {
      // Backward iteration - use the first element.
      stop = std::numeric_limits<int>::lowest();
    }
  }

  // Handle negative indices
  const int axis_size = input_shape.Dims(axis);
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

inline bool LoopCondition(int index, int stop, int stride)
{
  // True when we have reached the end of an axis and should loop.
  return stride > 0 ? index >= stop : index <= stop;
}

template <typename T>
inline StridedSliceParams
buildStridedSliceParams(const T *begin, const T *end, const T *strides, const uint32_t begin_mask,
                        const uint32_t end_mask, const uint32_t shrink_axis_mask,
                        const uint8_t rank)
{
  StridedSliceParams op_params;
  op_params.start_indices_count = rank;
  op_params.stop_indices_count = rank;
  op_params.strides_count = rank;

  for (int i = 0; i < rank; ++i)
  {
    op_params.start_indices[i] = begin[i];
    op_params.stop_indices[i] = end[i];
    op_params.strides[i] = strides[i];

    assert(op_params.strides[i] != 0);
  }

  op_params.begin_mask = begin_mask;
  op_params.ellipsis_mask = 0; // NYI
  op_params.end_mask = end_mask;
  op_params.new_axis_mask = 0; // NYI
  op_params.shrink_axis_mask = shrink_axis_mask;

  assert(sizeof(op_params.begin_mask) * 4 >= rank);

  return op_params;
}

void checkOutputSize(const StridedSliceParams &op_params, const Shape &input_shape,
                     const Shape &output_shape, uint32_t rank)
{
  UNUSED_RELEASE(output_shape);

  int32_t shape_size = 0;

  for (uint32_t idx = 0; idx < rank; ++idx)
  {
    int32_t stride = op_params.strides[idx];
    int32_t begin = StartForAxis(op_params, input_shape, idx);
    int32_t end = StopForAxis(op_params, input_shape, idx, begin);

    // When shrinking an axis, the end position does not matter (and can be
    // incorrect when negative indexing is used, see Issue #19260). Always use
    // begin + 1 to generate a length 1 slice, since begin has
    // already been adjusted for negative indices by StartForAxis.
    const bool shrink_axis = op_params.shrink_axis_mask & (1 << idx);
    if (shrink_axis)
    {
      end = begin + 1;
    }

    int32_t dim_shape = std::ceil((end - begin) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis)
    {
      assert(output_shape.Dims(shape_size) == dim_shape);
      shape_size++;
    }
  }

  assert(output_shape.DimensionsCount() == shape_size);
}

template <typename T>
inline void StridedSlice(const StridedSliceParams &op_params, const Shape &unextended_input_shape,
                         const T *input_data, const Shape &unextended_output_shape, T *output_data)
{
  assert(unextended_input_shape.DimensionsCount() <= 4);
  assert(unextended_output_shape.DimensionsCount() <= 4);

  bool optimize = true;
  int st_count = op_params.strides_count;
  for (int idx = 0; idx < st_count - 1; idx++)
  {
    const int axis_size = unextended_input_shape.Dims(idx);
    const int start = StartForAxis(op_params, unextended_input_shape, idx);
    const int stop = StopForAxis(op_params, unextended_input_shape, idx, start);
    if ((axis_size != 1) && (start != 0 || stop != 0))
    {
      optimize = false;
      break;
    }
  }

  if (optimize)
  {
    if (op_params.strides[st_count - 1] == 1)
    {
      const int start = StartForAxis(op_params, unextended_input_shape, st_count - 1);
      const int end = StopForAxis(op_params, unextended_input_shape, st_count - 1, start);

      for (int idx = 0; idx < end - start; idx++)
      {
        output_data[idx] = input_data[idx + start];
      }
      return;
    }
  }

  // Note that the output_shape is not used herein.
  StridedSliceParams params_copy = op_params;

  const Shape input_shape = Shape::ExtendedShape(4, unextended_input_shape);
  const Shape output_shape = Shape::ExtendedShape(4, unextended_output_shape);

  // Reverse and pad to 4 dimensions because that is what the runtime code
  // requires (ie. all shapes must be 4D and are given backwards).
  StridedSlicePadIndices(&params_copy, 4);

  const int start_b = StartForAxis(params_copy, input_shape, 0);
  const int stop_b = StopForAxis(params_copy, input_shape, 0, start_b);
  const int start_h = StartForAxis(params_copy, input_shape, 1);
  const int stop_h = StopForAxis(params_copy, input_shape, 1, start_h);
  const int start_w = StartForAxis(params_copy, input_shape, 2);
  const int stop_w = StopForAxis(params_copy, input_shape, 2, start_w);
  const int start_d = StartForAxis(params_copy, input_shape, 3);
  const int stop_d = StopForAxis(params_copy, input_shape, 3, start_d);

  T *out_ptr = output_data;
  for (int in_b = start_b; !LoopCondition(in_b, stop_b, params_copy.strides[0]);
       in_b += params_copy.strides[0])
  {
    for (int in_h = start_h; !LoopCondition(in_h, stop_h, params_copy.strides[1]);
         in_h += params_copy.strides[1])
    {
      for (int in_w = start_w; !LoopCondition(in_w, stop_w, params_copy.strides[2]);
           in_w += params_copy.strides[2])
      {
        for (int in_d = start_d; !LoopCondition(in_d, stop_d, params_copy.strides[3]);
             in_d += params_copy.strides[3])
        {
          *out_ptr++ = input_data[Offset(input_shape, in_b, in_h, in_w, in_d)];
        }
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_STRIDEDSLICE_H__
