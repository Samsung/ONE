/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "OperationUtils.h"
#include "ResizeBilinearLayer.h"
#include "cker/operation/ResizeBilinear.h"
#include <cker/Types.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

ResizeBilinearLayer::ResizeBilinearLayer()
  : _input(nullptr), _output(nullptr), _size(nullptr), _output_height(0), _output_width(0),
    _align_corners(false), _half_pixel_centers(false)
{
  // DO NOTHING
}

void ResizeBilinearLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                                    const IPortableTensor *size, bool align_corners,
                                    bool half_pixel_centers)
{
  assert(!size->is_constant());
  _input = input;
  _output = output;
  _size = size;
  _align_corners = align_corners;
  _half_pixel_centers = half_pixel_centers;
}

void ResizeBilinearLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                                    int32_t output_height, int32_t output_width, bool align_corners,
                                    bool half_pixel_centers)
{
  assert(_size == nullptr);
  if (output_height < 0)
  {
    throw std::runtime_error{"ResizeBilinear: size value must be positive value, output_height = " +
                             std::to_string(output_height)};
  }
  if (output_width < 0)
  {
    throw std::runtime_error{"ResizeBilinear: size value must be positive value, output_width = " +
                             std::to_string(output_width)};
  }
  _input = input;
  _output = output;
  _output_height = output_height;
  _output_width = output_width;
  _align_corners = align_corners;
  _half_pixel_centers = half_pixel_centers;
}

void ResizeBilinearLayer::run()
{
  nnfw::cker::ResizeBilinearParams params;
  if (_size == nullptr)
  {
    params.output_height = _output_height;
    params.output_width = _output_width;
  }
  else
  {
    const auto size_buf = getBuffer<int32_t>(_size);
    params.output_height = size_buf[0];
    params.output_width = size_buf[1];
  }
  params.align_corners = _align_corners;
  params.half_pixel_centers = _half_pixel_centers;

  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::ResizeBilinear(params, getShape(_input), getBuffer<float>(_input),
                                 getShape(_output), getBuffer<float>(_output));
      break;

    case OperandType::QUANT_UINT8_ASYMM:
      nnfw::cker::ResizeBilinear(params, getShape(_input), getBuffer<uint8_t>(_input),
                                 getShape(_output), getBuffer<uint8_t>(_output));
      break;

    case OperandType::QUANT_INT8_ASYMM:
      nnfw::cker::ResizeBilinear(params, getShape(_input), getBuffer<int8_t>(_input),
                                 getShape(_output), getBuffer<int8_t>(_output));
      break;

    case OperandType::UINT8:
    case OperandType::BOOL8:
    case OperandType::FLOAT16:
    case OperandType::INT32:
    case OperandType::INT64:
    case OperandType::QUANT_INT8_SYMM:
      std::runtime_error("ResizeBilinear NYI");
      break;
    default:
      std::runtime_error("ResizeBilinear unsupported data type");
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
