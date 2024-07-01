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

IOTensor::IOTensor(const ir::OperandInfo &info, ir::Layout layout)
  : IPortableTensor{info}, _is_dynamic{false}, _tensor{nullptr},
    _orig{std::make_unique<UserTensor>(info, layout, (uint8_t *)nullptr, 0)}
{
  _tensor = _orig.get();
}

void IOTensor::setTensor(IPortableTensor *tensor)
{
  assert(tensor);
  assert(tensor != this);
  // TODO Handle when layout was changed
  assert(tensor->layout() == _orig->layout()); // Changing layout is not considered yet
  _tensor = tensor;
  if (_orig->getShape() != tensor->getShape())
  {
    _orig->setShape(tensor->getShape());
    _orig->set_dynamic();
    _is_dynamic = true;
  }
  else
    _is_dynamic = false;
}

} // namespace builtin
} // namespace backend
} // namespace onert
