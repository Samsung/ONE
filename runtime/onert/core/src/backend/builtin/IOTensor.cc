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

#include "IOTensor.h"

#include <assert.h>

namespace onert
{
namespace backend
{
namespace builtin
{

// `dynamic_cast` not working across library boundaries on NDK
// With this as a key function, `dynamic_cast` works across dl
IOTensor::~IOTensor() {}

IOTensor::IOTensor(const ir::OperandInfo &info) : IPortableTensor{info}, _orig_info{info}
{
  setUserTensor(nullptr, 0);
}

void IOTensor::setTensor(IPortableTensor *tensor)
{
  assert(tensor);
  assert(tensor != this);
  // TODO Handle when layout was changed
  assert(tensor->layout() == _orig_info.layout()); // Changing layout is not considered yet
  _user_tensor.reset();
  _tensor = tensor;
}

void IOTensor::setUserTensor(uint8_t *buffer, size_t size)
{
  _user_tensor = std::make_unique<UserTensor>(_orig_info, buffer, size);
  _tensor = _user_tensor.get();
}

} // namespace builtin
} // namespace backend
} // namespace onert
