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
namespace cpu
{
namespace kernel
{

StridedSliceLayer::StridedSliceLayer()
    : _input(nullptr), _begin(nullptr), _end(nullptr), _strides(nullptr), _output(nullptr),
      _begin_mask(0), _ellipsis_mask(0), _end_mask(0), _new_axis_mask(0), _shrink_axis_mask(0),
      _rank(0)
{
}

void StridedSliceLayer::stridedSliceFloat32()
{
  auto op_params = nnfw::cker::buildStridedSliceParams(
      reinterpret_cast<uint32_t *>(_begin->buffer()), reinterpret_cast<uint32_t *>(_end->buffer()),
      reinterpret_cast<uint32_t *>(_strides->buffer()), _begin_mask, _end_mask, _shrink_axis_mask,
      _rank);

  nnfw::cker::checkOutputSize(op_params, convertTensorToCkerShape(_input),
                              convertTensorToCkerShape(_output), _rank);

  nnfw::cker::StridedSlice(op_params, convertTensorToCkerShape(_input),
                           reinterpret_cast<const float *>(_input->buffer()),
                           convertTensorToCkerShape(_output),
                           reinterpret_cast<float *>(_output->buffer()));
}

void StridedSliceLayer::stridedSliceQuant8()
{
  // cker quant8 stridedSlice is not implemented yet
  throw std::runtime_error{"NYI"};
}

void StridedSliceLayer::configure(const ITensor *input, const ITensor *begin, const ITensor *end,
                                  const ITensor *strides, ITensor *output, const int32_t begin_mask,
                                  const int32_t end_mask, const int32_t shrink_axis_mask,
                                  const int32_t rank)
{
  _input = input;
  _begin = begin;
  _end = end;
  _strides = strides;
  _output = output;

  _rank = rank;
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
    stridedSliceFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT8_ASYMM)
  {
    stridedSliceQuant8();
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
