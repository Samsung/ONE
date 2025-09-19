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

#include "DynamicUpdateSliceLayer.h"
#include "OperationUtils.h"

#include <cker/operation/DynamicUpdateSlice.h>
#include <util/Exceptions.h>

namespace onert::backend::cpu::ops
{

DynamicUpdateSliceLayer::DynamicUpdateSliceLayer()
  : _operand(nullptr), _update(nullptr), _indices(nullptr), _output(nullptr)
{
  // DO NOTHING
}

DynamicUpdateSliceLayer::~DynamicUpdateSliceLayer() = default;

void DynamicUpdateSliceLayer::configure(const IPortableTensor *operand,
                                        const IPortableTensor *update,
                                        const IPortableTensor *indices, IPortableTensor *output)
{
  assert(operand != nullptr);
  assert(update != nullptr);
  assert(indices != nullptr);
  assert(output != nullptr);

  _operand = operand;
  _update = update;
  _indices = indices;
  _output = output;
}

void DynamicUpdateSliceLayer::run()
{
  // Get indices data as int64 type vector
  std::vector<int64_t> indices_data(_indices->getShape().num_elements());
  for (size_t i = 0; i < indices_data.size(); ++i)
  {
    if (_indices->data_type() == OperandType::INT32)
    {
      indices_data[i] = static_cast<int64_t>(getBuffer<int32_t>(_indices)[i]);
    }
    else
    {
      assert(_indices->data_type() == OperandType::INT64);
      indices_data[i] = getBuffer<int64_t>(_indices)[i];
    }
  }

  switch (_operand->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::DynamicUpdateSlice()(getShape(_operand), getBuffer<float>(_operand),
                                       getShape(_update), getBuffer<float>(_update), indices_data,
                                       getBuffer<float>(_output));
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      nnfw::cker::DynamicUpdateSlice()(getShape(_operand), getBuffer<uint8_t>(_operand),
                                       getShape(_update), getBuffer<uint8_t>(_update), indices_data,
                                       getBuffer<uint8_t>(_output));
      break;
    case OperandType::QUANT_INT16_SYMM:
      nnfw::cker::DynamicUpdateSlice()(getShape(_operand), getBuffer<int8_t>(_operand),
                                       getShape(_update), getBuffer<int8_t>(_update), indices_data,
                                       getBuffer<int8_t>(_output));
      break;
    default:
      throw UnsupportedDataTypeException{"DynamicUpdateSlice", _operand->data_type()};
      break;
  }
}

} // namespace onert::backend::cpu::ops
