/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PadLayer.h"

#include <cker/operation/Pad.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

PadLayer::PadLayer()
  : _input(nullptr), _pad(nullptr), _value(nullptr), _output(nullptr), _constantValueData()
{
  // DO NOTHING
}

template <typename T> void PadLayer::padImpl(const T *constant_value_data)
{
  assert(_pad->data_type() == onert::ir::DataType::INT32);
  assert(_pad->buffer());
  const auto pad_data = reinterpret_cast<const int32_t *>(_pad->buffer());
  auto pad_rank = _pad->getShape().dim(0);
  nnfw::cker::Pad<T>(pad_data, pad_rank, getShape(_input), getBuffer<T>(_input), getShape(_output),
                     getBuffer<T>(_output), constant_value_data);
}

void PadLayer::configure(const IPortableTensor *input, const IPortableTensor *pad,
                         const IPortableTensor *value, IPortableTensor *output)
{
  _input = input;
  _pad = pad;
  _value = value;
  _output = output;
}

void PadLayer::run()
{
  if (_value != nullptr) // isPadV2
  {
    assert(_value->buffer());
    _constantValueData.v = reinterpret_cast<const void *>(_value->buffer());
  }

  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      padImpl<float>(_constantValueData.f);
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      if (_constantValueData.u8 == nullptr)
      {
        uint8_t pad_value = static_cast<uint8_t>(_output->data_zero_point());
        padImpl<uint8_t>(&pad_value);
      }
      else
      {
        padImpl<uint8_t>(_constantValueData.u8);
      }
      break;
    case OperandType::QUANT_INT8_ASYMM:
      if (_constantValueData.i8 == nullptr)
      {
        int8_t pad_value = static_cast<int8_t>(_output->data_zero_point());
        padImpl<int8_t>(&pad_value);
      }
      else
      {
        padImpl<int8_t>(_constantValueData.i8);
      }
      break;
    default:
      throw std::runtime_error{"Pad: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
