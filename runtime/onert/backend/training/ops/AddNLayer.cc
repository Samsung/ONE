/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
namespace training
{
namespace ops
{

void AddNLayer::configure(std::vector<const IPortableTensor *> &&inputs, IPortableTensor *output)
{
  _inputs = std::move(inputs);
  _output = output;
}

void AddNLayer::run()
{
  size_t input_size = _inputs.size();
  if (_output->data_type() == ir::DataType::INT32)
  {
    std::vector<const int32_t *> input_buffers(input_size);
    for (size_t i = 0; i < input_size; i++)
    {
      input_buffers[i] = getBuffer<int32_t>(_inputs[i]);
    }
    AddN(getShape(_inputs[0]), input_size, input_buffers.data(), getBuffer<int32_t>(_output));
  }
  else if (_output->data_type() == ir::DataType::FLOAT32)
  {
    std::vector<const float *> input_buffers(input_size);
    for (size_t i = 0; i < input_size; i++)
    {
      input_buffers[i] = getBuffer<float>(_inputs[i]);
    }
    AddN(getShape(_inputs[0]), input_size, input_buffers.data(), getBuffer<float>(_output));
  }
  else
  {
    throw std::runtime_error("AddN: unsupported data type");
  }
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
