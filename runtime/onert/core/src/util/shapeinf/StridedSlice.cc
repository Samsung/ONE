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

#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{
struct StridedSliceParams
{
  int8_t start_indices_count;
  int16_t start_indices[4];
  int8_t stop_indices_count;
  int16_t stop_indices[4];
  int8_t strides_count;
  int16_t strides[4];

  int16_t begin_mask;
  int16_t ellipsis_mask;
  int16_t end_mask;
  int16_t new_axis_mask;
  int16_t shrink_axis_mask;
};

int Clamp(const int v, const int lo, const int hi)
{
  assert(!(hi < lo));
  if (hi < v)
    return hi;
  if (v < lo)
    return lo;
  return v;
}

int StartForAxis(const StridedSliceParams &params, const ir::Shape &input_shape, int axis)
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
  int axis_size = input_shape.dim(axis);
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
int StopForAxis(const StridedSliceParams &params, const ir::Shape &input_shape, int axis,
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

  const int axis_size = input_shape.dim(axis);
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

template <typename T>
StridedSliceParams buildStridedSliceParams(const T *begin, const T *end, const T *strides,
                                           const uint32_t begin_mask, const uint32_t end_mask,
                                           const uint32_t shrink_axis_mask, const uint8_t rank)
{
  StridedSliceParams op_params;
  op_params.start_indices_count = rank;
  op_params.stop_indices_count = rank;
  op_params.strides_count = rank;

  for (int i = 0; i < op_params.strides_count; ++i)
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

ir::Shape inferStridedSliceShape(const ir::Shape &input_shape, const StridedSliceParams &op_params,
                                 uint32_t rank)
{
  ir::Shape out_shape;
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
      shape_size++;
      out_shape.append(dim_shape);
    }
  }

  return out_shape;
}

void StaticInferer::visit(const ir::operation::StridedSlice &op)
{

  const auto input_index{op.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto &input = _operands.at(input_index);
  const auto starts_index{op.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto &starts = _operands.at(starts_index);
  const auto ends_index{op.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto &ends = _operands.at(ends_index);
  const auto strides_index{op.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};
  const auto &strides = _operands.at(strides_index);
  const auto output_index = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_index);

  if (input.info().isDynamic() || starts.info().isDynamic() || ends.info().isDynamic() ||
      strides.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  if (!(starts.isConstant() && ends.isConstant() && strides.isConstant()))
  {
    output.info().setDynamic();
    return;
  }

  const auto begin_mask = op.param().begin_mask;
  const auto end_mask = op.param().end_mask;
  const auto shrink_axis_mask = op.param().shrink_axis_mask;
  const auto rank = op.param().rank;

  auto starts_buf = reinterpret_cast<const uint32_t *>(starts.data()->base());
  auto ends_buf = reinterpret_cast<const uint32_t *>(ends.data()->base());
  auto strides_buf = reinterpret_cast<const uint32_t *>(strides.data()->base());

  auto op_params = buildStridedSliceParams(starts_buf, ends_buf, strides_buf, begin_mask, end_mask,
                                           shrink_axis_mask, rank);

  ir::Shape new_shape = inferStridedSliceShape(input.info().shape(), op_params, rank);
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::StridedSlice &op)
{

  const auto input_index{op.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  auto input = _tensor_registry->getITensor(input_index);
  ir::Shape input_shape = getShape(input.get());

  const auto starts_index{op.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  auto starts = _tensor_registry->getITensor(starts_index);

  const auto ends_index{op.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  auto ends = _tensor_registry->getITensor(ends_index);

  const auto strides_index{op.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};
  auto strides = _tensor_registry->getITensor(strides_index);

  if (!(input->is_dynamic() || starts->is_dynamic() || ends->is_dynamic() || strides->is_dynamic()))
  {
    return;
  }

  const auto begin_mask = op.param().begin_mask;
  const auto end_mask = op.param().end_mask;
  const auto shrink_axis_mask = op.param().shrink_axis_mask;
  const auto rank = input_shape.rank();

  auto op_params = buildStridedSliceParams(reinterpret_cast<uint32_t *>(starts->buffer()),
                                           reinterpret_cast<uint32_t *>(ends->buffer()),
                                           reinterpret_cast<uint32_t *>(strides->buffer()),
                                           begin_mask, end_mask, shrink_axis_mask, rank);

  auto output_index = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_index);

  ir::Shape output_shape =
      onert::shape_inference::inferStridedSliceShape(input_shape, op_params, rank);

  _dynamic_tensor_manager->applyShape(output_index, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
