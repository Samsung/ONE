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

PadLayer::PadLayer() : cpu::ops::PadLayer(), _back_prop_input{nullptr}, _back_prop_output{nullptr}
{
  // DO NOTHING
}

template <typename T> void PadLayer::depad()
{
  nnfw::cker::train::Depad<T>(_padData, _padRank, getShape(_back_prop_output),
                              getBuffer<T>(_back_prop_output), getShape(_back_prop_input),
                              getBuffer<T>(_back_prop_input));
}

void PadLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                         const int32_t *padData, int32_t padRank, const void *constantValueData,
                         IPortableTensor *back_prop_input, const IPortableTensor *back_prop_output)
{
  cpu::ops::PadLayer::configure(input, output, padData, padRank, constantValueData);
  _back_prop_input = back_prop_input;
  _back_prop_output = back_prop_output;
}

void PadLayer::forward(bool) { cpu::ops::PadLayer::run(); }

void PadLayer::backward()
{
  switch (_back_prop_output->data_type())
  {
    case OperandType::FLOAT32:
      depad<float>();
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      depad<uint8_t>();
      break;
    case OperandType::QUANT_INT8_ASYMM:
      depad<int8_t>();
      break;
    default:
      throw std::runtime_error{"Pad: unsupported data type"};
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
