/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "StridedSliceLayer.h"

#include "OperationUtils.h"

#include <cker/operation/StridedSlice.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

StridedSliceLayer::StridedSliceLayer()
  : _input(nullptr), _begin(nullptr), _end(nullptr), _strides(nullptr), _output(nullptr),
    _begin_mask(0), _ellipsis_mask(0), _end_mask(0), _new_axis_mask(0), _shrink_axis_mask(0)
{
}

template <typename T> void StridedSliceLayer::stridedSliceImpl()
{
  const auto input_shape = getShape(_input);
  const auto output_shape = getShape(_output);
  auto op_params = nnfw::cker::buildStridedSliceParams(
    getBuffer<uint32_t>(_begin), getBuffer<uint32_t>(_end), getBuffer<uint32_t>(_strides),
    _begin_mask, _end_mask, _shrink_axis_mask, input_shape.DimensionsCount());

  nnfw::cker::checkOutputSize(op_params, input_shape, output_shape, input_shape.DimensionsCount());

  nnfw::cker::StridedSlice(op_params, input_shape, getBuffer<T>(_input), output_shape,
                           getBuffer<T>(_output));
}

void StridedSliceLayer::configure(const IPortableTensor *input, const IPortableTensor *begin,
                                  const IPortableTensor *end, const IPortableTensor *strides,
                                  IPortableTensor *output, const int32_t begin_mask,
                                  const int32_t end_mask, const int32_t shrink_axis_mask)
{
  _input = input;
  _begin = begin;
  _end = end;
  _strides = strides;
  _output = output;

  _begin_mask = begin_mask;
  _ellipsis_mask = 0;
  _end_mask = end_mask;
  _new_axis_mask = 0;
  _shrink_axis_mask = shrink_axis_mask;
}

void StridedSliceLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    stridedSliceImpl<float>();
  }
  else if (_input->data_type() == OperandType::INT32)
  {
    stridedSliceImpl<int32_t>();
  }
  else if (_input->data_type() == OperandType::INT64)
  {
    stridedSliceImpl<int64_t>();
  }
  else
  {
    throw std::runtime_error{"StridedSlice: unsupported data type"};
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
