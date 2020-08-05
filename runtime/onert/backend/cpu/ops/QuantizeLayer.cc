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

#include "QuantizeLayer.h"

#include <cker/operation/Quantize.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

QuantizeLayer::QuantizeLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

template <typename InputT, typename OutputT> void QuantizeLayer::affineQuantize()
{
  nnfw::cker::Quantize(getTensorShape(_input), reinterpret_cast<const InputT *>(_input->buffer()),
                       getTensorShape(_output), reinterpret_cast<OutputT *>(_output->buffer()),
                       _output->data_scale(), _output->data_offset());
}

void QuantizeLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void QuantizeLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    affineQuantize<float, uint8_t>();
  }
  else
  {
    throw std::runtime_error{"Quantize: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
