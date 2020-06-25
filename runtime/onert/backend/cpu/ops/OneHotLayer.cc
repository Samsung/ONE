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

#include "OneHotLayer.h"

#include "OperationUtils.h"

#include <cker/operation/OneHot.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

void OneHotLayer::oneHotFloat32()
{
  nnfw::cker::OneHot<float, int32_t>(_depth, _on_value, _off_value, _axis, getTensorShape(_indices),
                                     reinterpret_cast<const int32_t *>(_indices->buffer()),
                                     getTensorShape(_output),
                                     reinterpret_cast<float *>(_output->buffer()));
}

void OneHotLayer::oneHotQuant8() { throw std::runtime_error{"OneHot NYI for quantized"}; }

void OneHotLayer::configure(const IPortableTensor *indices, IPortableTensor *output, int32_t depth,
                            float on_value, float off_value, int32_t axis)
{
  _indices = indices;
  _output = output;
  _depth = depth;
  _on_value = on_value;
  _off_value = off_value;
  _axis = axis;
}

void OneHotLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    oneHotFloat32();
  }
  else if (_output->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    oneHotQuant8();
  }
  else
  {
    throw std::runtime_error{"OneHot: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
