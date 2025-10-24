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

#include "RoPELayer.h"

#include <cker/operation/RoPE.h>
#include <util/Exceptions.h>

namespace onert::backend::cpu::ops
{

RoPELayer::RoPELayer()
  : _input(nullptr), _sin(nullptr), _cos(nullptr), _mode(nnfw::cker::RoPEMode::kGptNeox),
    _output(nullptr)
{
  // DO NOTHING
}

RoPELayer::~RoPELayer() = default;

void RoPELayer::configure(const IPortableTensor *input, const IPortableTensor *sin,
                          const IPortableTensor *cos, nnfw::cker::RoPEMode mode,
                          IPortableTensor *output)
{
  assert(input != nullptr);
  assert(sin != nullptr);
  assert(cos != nullptr);
  assert(output != nullptr);

  _input = input;
  _sin = sin;
  _cos = cos;
  _mode = mode;
  _output = output;
}

template <typename T> void RoPELayer::rope()
{
  auto input_shape = _input->getShape();
  assert(input_shape.rank() == 4);

  nnfw::cker::RoPE(_mode, getShape(_input), getBuffer<T>(_input), getShape(_sin),
                   getBuffer<T>(_sin), getShape(_cos), getBuffer<T>(_cos), getShape(_output),
                   getBuffer<T>(_output));
}

void RoPELayer::run()
{
  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      rope<float>();
      break;
    case OperandType::INT32:
      rope<int32_t>();
      break;
    case OperandType::INT64:
      rope<int64_t>();
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      rope<uint8_t>();
      break;
    case OperandType::QUANT_INT8_ASYMM:
      rope<int8_t>();
      break;
    default:
      throw UnsupportedDataTypeException{"RoPE", _input->data_type()};
  }
}

} // namespace onert::backend::cpu::ops
