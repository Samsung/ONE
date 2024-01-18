/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/train/operation/Pad.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

PadLayer::PadLayer()
  : _input(nullptr), _output(nullptr), _padData(), _padRank(), _constantValueData(),
    _back_prop_input{nullptr}, _back_prop_output{nullptr}
{
  // DO NOTHING
}

template <typename T> void PadLayer::padImpl(const T *constant_value_data)
{
  // TODO: Use nnfw::cker::Pad
  nnfw::cker::train::Pad<T>(_padData, _padRank, getShape(_input), getBuffer<T>(_input), getShape(_output),
                       getBuffer<T>(_output), constant_value_data);
}

void PadLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                         const int32_t *padData, int32_t padRank, const void *constantValueData,
                         IPortableTensor *back_prop_input, const IPortableTensor *back_prop_output)
{
  _input = input;
  _output = output;
  memcpy(_padData, padData, sizeof(_padData));
  _padRank = padRank;
  _constantValueData.v = constantValueData;
  _back_prop_input = back_prop_input;
  _back_prop_output = back_prop_output;
}

void PadLayer::forward(bool)
{
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

void PadLayer::backward()
{
  // DO NOTHING
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
