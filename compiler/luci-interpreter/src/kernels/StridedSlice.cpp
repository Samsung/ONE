/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/StridedSlice.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

const int max_dim = 4;

// Use until std::clamp() is available from C++17.
inline int Clamp(const int v, const int lo, const int hi)
{
  TFLITE_DCHECK(!(hi < lo));
  if (hi < v)
    return hi;
  if (v < lo)
    return lo;
  return v;
}

inline int32_t positiveRemainder(int32_t dividend, int32_t divisor)
{
  return (divisor + (dividend % divisor)) % divisor;
}

inline int32_t clampedIndex(int32_t index, int dim, bool pos_stride)
{
  return pos_stride
             ? (index >= dim ? dim : positiveRemainder(std::min(std::max(index, -dim), dim), dim))
             : (index < -dim ? -1
                             : positiveRemainder(std::min(std::max(index, -dim), dim - 1), dim));
}

StridedSlice::StridedSlice(const Tensor *input, const Tensor *begin, const Tensor *end,
                           const Tensor *strides, Tensor *output, const StridedSliceParams &params)
    : KernelWithParams<StridedSliceParams>(params), _input(input), _begin(begin), _end(end),
      _strides(strides), _output(output)
{
}

void StridedSlice::configure()
{
  assert(_begin->shape().num_dims() == 1);
  assert(_end->shape().num_dims() == 1);
  assert(_strides->shape().num_dims() == 1);
  assert(_input->element_type() == _output->element_type());
  assert(_begin->element_type() == DataType::S32);
  assert(_end->element_type() == DataType::S32);
  assert(_strides->element_type() == DataType::S32);
  assert(_input->shape().num_dims() <= max_dim);
  assert(params().ellipsis_mask == 0);
  assert(params().new_axis_mask == 0);
  if (_input->element_type() == DataType::U8)
  {
    assert(_input->scale() == _output->scale());
    assert(_input->zero_point() == _output->zero_point());
  }
  std::vector<int32_t> output_shape_vector;
  for (int i = 0; i < _input->shape().num_dims(); i++)
  {
    int idx = _input->shape().num_dims() - i - 1;
    int32_t stride_value = getTensorData<int32_t>(_strides)[idx];
    assert(stride_value != 0);
    bool pos_stride = stride_value > 0;
    int dim = _input->shape().dim(idx);

    const auto begin_mask = params().begin_mask;
    int32_t start = getTensorData<int32_t>(_begin)[idx];
    if (begin_mask & 1 << idx)
    {
      if (pos_stride)
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
    if (start < 0)
    {
      start += dim;
    }
    start = Clamp(start, 0, dim - 1);

    const auto end_mask = params().end_mask;
    const auto shrink_axis_mask = params().shrink_axis_mask;
    const bool shrink_axis = shrink_axis_mask & (1 << idx);
    int32_t end = getTensorData<int32_t>(_end)[idx];

    if (shrink_axis)
    {
      end = start + 1;
    }

    // end_mask override
    if (end_mask & (1 << idx))
    {
      if (pos_stride)
      {
        // Forward iteration - use the last element. These values will get
        // clamped below
        end = std::numeric_limits<int>::max();
      }
      else
      {
        // Backward iteration - use the first element.
        end = std::numeric_limits<int>::lowest();
      }
    }
    if (end < 0)
    {
      end += dim;
    }

    // Clamping
    // Because the end index points one past the last element, we need slightly
    // different clamping ranges depending on the direction.
    if (pos_stride)
    {
      // Forward iteration
      end = Clamp(end, 0, dim);
    }
    else
    {
      // Backward iteration
      end = Clamp(end, -1, dim - 1);
    }

    if (shrink_axis)
    {
      end = start + 1;
    }
    int32_t dim_shape = std::ceil((end - start) / static_cast<float>(stride_value));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis)
    {
      output_shape_vector.push_back(dim_shape);
    }
  }
  Shape output_shape = Shape(output_shape_vector.size());
  for (int32_t i = 0; i < output_shape_vector.size(); i++)
  {
    output_shape.dim(i) = output_shape_vector[output_shape_vector.size() - i - 1];
  }
  _output->resize(output_shape);
}

void StridedSlice::execute() const
{
  tflite::StridedSliceParams op_params;
  op_params.start_indices_count = _input->shape().num_dims();
  op_params.stop_indices_count = _input->shape().num_dims();
  op_params.strides_count = _input->shape().num_dims();

  for (int i = 0; i < _input->shape().num_dims(); i++)
  {
    op_params.start_indices[i] = getTensorData<int32_t>(_begin)[i];
    op_params.stop_indices[i] = getTensorData<int32_t>(_end)[i];
    op_params.strides[i] = getTensorData<int32_t>(_strides)[i];
  }
  op_params.begin_mask = params().begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = params().end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = params().shrink_axis_mask;

  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      tflite::reference_ops::StridedSlice(op_params, getTensorShape(_input),
                                          getTensorData<float>(_input), getTensorShape(_output),
                                          getTensorData<float>(_output));
      break;
    case DataType::U8:
      tflite::reference_ops::StridedSlice(op_params, getTensorShape(_input),
                                          getTensorData<uint8_t>(_input), getTensorShape(_output),
                                          getTensorData<uint8_t>(_output));
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
