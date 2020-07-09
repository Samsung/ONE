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

const int kMaxDim = 4;

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
  assert(_input->shape().num_dims() <= kMaxDim);
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
    int _idx = _input->shape().num_dims() - i - 1;
    int32_t _stride_value = getTensorData<int32_t>(_strides)[_idx];
    assert(_stride_value != 0);
    bool _pos_stride = _stride_value > 0;
    int _dim = _input->shape().dim(_idx);

    int32_t _begin_value =
        params().begin_mask & (1 << _idx)
            ? _pos_stride ? 0 : _dim - 1
            : clampedIndex(getTensorData<int32_t>(_begin)[_idx], _dim, _pos_stride);
    int32_t _end_value = params().end_mask & (1 << _idx)
                             ? _pos_stride ? 0 : _dim - 1
                             : clampedIndex(getTensorData<int32_t>(_end)[_idx], _dim, _pos_stride);
    bool _shrink_axis = params().shrink_axis_mask & (1 << _idx);
    if (_shrink_axis)
    {
      _end_value = _begin_value + 1;
    }
    int32_t _dim_shape = std::ceil((_end_value - _begin_value) / static_cast<float>(_stride_value));
    _dim_shape = _dim_shape < 0 ? 0 : _dim_shape;
    if (!_shrink_axis)
    {
      output_shape_vector.push_back(_dim_shape);
    }
  }
  Shape output_shape = Shape(output_shape_vector.size());
  for (int i = 0; i < output_shape_vector.size(); i++)
  {
    output_shape.dim(i) = output_shape_vector[output_shape_vector.size() - i - 1];
  }
  _output->resize(output_shape);
}

void StridedSlice::execute() const
{
  std::vector<int32_t> starts;
  std::vector<int32_t> stops;
  std::vector<int32_t> strides;
  for (int i = _input->shape().num_dims(); i < kMaxDim; i++)
  {
    starts.emplace_back(0);
    stops.emplace_back(1);
    strides.emplace_back(1);
  }
  for (int idx = 0; idx < _input->shape().num_dims(); ++idx)
  {
    starts.emplace_back(getTensorData<int32_t>(_begin)[idx]);
    stops.emplace_back(getTensorData<int32_t>(_end)[idx]);
    strides.emplace_back(getTensorData<int32_t>(_strides)[idx]);
  }
  int begin_mask = params().begin_mask << (4 - _input->shape().num_dims());
  int end_mask = params().end_mask << (4 - _input->shape().num_dims());
  int shrink_axis_mask = params().shrink_axis_mask << (4 - _input->shape().num_dims());

  assert(starts.size() == 4);
  tflite::StridedSliceParams op_params = ::tflite::strided_slice::BuildStridedSliceParams(
      begin_mask, end_mask, shrink_axis_mask, starts, stops, strides);

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
