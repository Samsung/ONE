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

#include "AddNLayer.h"

#include "OperationUtils.h"

#include <cker/operation/AddN.h>
#include <assert.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

void AddNLayer::configure(const IPortableTensor **inputs, size_t num_inputs,
                          IPortableTensor *output)
{
  _inputs = inputs;
  _num_inputs = num_inputs;
  _output = output;
}

void AddNLayer::run()
{
  if (_output->data_type() == ir::DataType::INT32)
  {
    const int32_t **inputs = (const int32_t **)malloc(sizeof(int32_t *) * _num_inputs);
    for (size_t i = 0; i < _num_inputs; i++)
    {
      inputs[i] = reinterpret_cast<int32_t *>(_inputs[i]->buffer());
    }
    AddN(getTensorShape(_inputs[0]), _num_inputs, reinterpret_cast<const int32_t **>(inputs),
         reinterpret_cast<int32_t *>(_output->buffer()));
  }
  else if (_output->data_type() == ir::DataType::FLOAT32)
  {
    const float **inputs = (const float **)malloc(sizeof(float *) * _num_inputs);
    for (size_t i = 0; i < _num_inputs; i++)
    {
      inputs[i] = reinterpret_cast<float *>(_inputs[i]->buffer());
    }
    AddN(getTensorShape(_inputs[0]), _num_inputs, reinterpret_cast<const float **>(inputs),
         reinterpret_cast<float *>(_output->buffer()));
  }
  else
  {
    throw std::runtime_error("AddN: unsupported data type");
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
