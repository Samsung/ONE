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

namespace onert::backend::builtin
{

// `dynamic_cast` not working across library boundaries on NDK
// With this as a key function, `dynamic_cast` works across dl
IOTensor::~IOTensor() {}

IOTensor::IOTensor(const ir::OperandInfo &info)
  : IPortableTensor{info}, _tensor{nullptr},
    _orig{std::make_unique<UserTensor>(info, (uint8_t *)nullptr, 0)}, _has_backend_tensor{false}
{
  _tensor = _orig.get();
}

void IOTensor::setTensor(IPortableTensor *tensor)
{
  assert(tensor);
  assert(tensor != this);
  assert(!_has_backend_tensor);
  _tensor = tensor;
  if (_info.shape() != tensor->getShape())
  {
    _info.shape(tensor->getShape());

    // If input tensor shape is updated, other effective buffers use dynamic memory manager.
    // Dynamic memory manager deallocate allcoated memory after each execution.
    // So we should remain input tensor as dynamic if we mark it dynamic at least once.
    // If dynamic memory manager maintains allocated memory after execution is finished,
    // we may need to reset it as static for each setTensor call.
    _info.setDynamic();
  }
}

void IOTensor::setBackendTensor(IPortableTensor *tensor)
{
  setTensor(tensor);
  _has_backend_tensor = true;
}

} // namespace onert::backend::builtin
