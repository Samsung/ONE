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
namespace kernel
{

PadLayer::PadLayer()
    : _input(nullptr), _output(nullptr), _padData(), _padRank(), _constantValueData()
{
  // DO NOTHING
}

void PadLayer::padFloat32()
{
  nnfw::cker::Pad(_padData, _padRank, convertTensorToCkerShape(_input),
                  reinterpret_cast<const float *>(_input->buffer()),
                  convertTensorToCkerShape(_output), reinterpret_cast<float *>(_output->buffer()),
                  _constantValueData.f);
}
void PadLayer::padQuant8() { throw std::runtime_error("Quantized Pad isn't supported NYI"); }

void PadLayer::configure(const operand::Tensor *input, operand::Tensor *output,
                         const int32_t *padData, int32_t padRank, uint8_t *constantValueData)
{
  _input = input;
  _output = output;
  memcpy(_padData, padData, sizeof(_padData));
  _padRank = padRank;
  _constantValueData.u8 = constantValueData;
}

void PadLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    padFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT8_ASYMM)
  {
    padQuant8();
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
