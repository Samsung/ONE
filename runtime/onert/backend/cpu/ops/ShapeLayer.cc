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

#include "ShapeLayer.h"

#include "OperationUtils.h"

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

ShapeLayer::ShapeLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

template <typename T> void GetRawShape(const IPortableTensor *input, T *output_data)
{
  for (uint32_t i = 0; i < input->num_dimensions(); ++i)
  {
    output_data[i] = static_cast<T>(input->dimension(i));
  }
}

void ShapeLayer::shape()
{
  if (_output->data_type() == OperandType::UINT32)
  {
    GetRawShape(_input, reinterpret_cast<uint32_t *>(_output->buffer()));
  }
  else if (_output->data_type() == OperandType::INT32)
  {
    GetRawShape(_input, reinterpret_cast<int32_t *>(_output->buffer()));
  }
  else
  {
    throw std::runtime_error{"NYI : not supported output type for ShapeLayer"};
  }
}

void ShapeLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void ShapeLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32 || _input->data_type() == OperandType::INT32 ||
      _input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    shape();
  }
  else
  {
    throw std::runtime_error{"Shape : unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
