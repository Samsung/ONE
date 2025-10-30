/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TopKV2Layer.h"

#include "OperationUtils.h"

#include <cker/operation/TopKV2.h>
#include <util/Exceptions.h>

namespace onert::backend::cpu::ops
{

TopKV2Layer::TopKV2Layer()
  : _input(nullptr), _output_value(nullptr), _output_indices(nullptr), _k(0)
{
}

void TopKV2Layer::configure(const IPortableTensor *input, IPortableTensor *output_value,
                            IPortableTensor *output_indices, const int32_t k)
{
  _input = input;
  _output_value = output_value;
  _output_indices = output_indices;

  _k = static_cast<uint32_t>(k);
}

void TopKV2Layer::run()
{
  if (_output_indices->data_type() != OperandType::INT32)
    throw UnsupportedDataTypeException{"TopKV2", _output_indices->data_type()};

  if (_input->data_type() == OperandType::FLOAT32)
  {
    nnfw::cker::TopKV2<float, int32_t>(getShape(_input), getBuffer<float>(_input), _k,
                                       getBuffer<float>(_output_value),
                                       getBuffer<int32_t>(_output_indices));
  }
  else if (_input->data_type() == OperandType::INT32)
  {
    nnfw::cker::TopKV2<int32_t, int32_t>(getShape(_input), getBuffer<int32_t>(_input), _k,
                                         getBuffer<int32_t>(_output_value),
                                         getBuffer<int32_t>(_output_indices));
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    nnfw::cker::TopKV2<uint8_t, int32_t>(getShape(_input), getBuffer<uint8_t>(_input), _k,
                                         getBuffer<uint8_t>(_output_value),
                                         getBuffer<int32_t>(_output_indices));
  }
  else
  {
    throw UnsupportedDataTypeException{"TopKV2", _input->data_type()};
  }
}

} // namespace onert::backend::cpu::ops
