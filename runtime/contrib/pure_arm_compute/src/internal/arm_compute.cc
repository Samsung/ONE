/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "internal/arm_compute.h"

#include <arm_compute/runtime/CL/CLScheduler.h>

#include <cassert>

namespace internal
{
namespace arm_compute
{
namespace operand
{

void Object::access(const std::function<void(::arm_compute::ITensor &tensor)> &fn) const
{
  if (::internal::arm_compute::isGpuMode())
  {
    auto &queue = ::arm_compute::CLScheduler::get().queue();

    auto cl_tensor = _tensor.get();
    CAST_CL(cl_tensor)->map(queue);
    fn(*_tensor);
    CAST_CL(cl_tensor)->unmap(queue);
  }
  else
  {
    fn(*_tensor);
  }
}

} // namespace operand
} // namespace arm_compute
} // namespace internal

namespace internal
{
namespace arm_compute
{
namespace operand
{

Context &Context::set(const ::internal::tflite::operand::Index &id,
                      const std::shared_ptr<::arm_compute::ITensor> &tensor)
{
  assert(_objects.find(id.asInt()) == _objects.end());

  _objects[id.asInt()] = Object{tensor};
  return (*this);
}

} // namespace operand
} // namespace arm_compute
} // namespace internal

namespace internal
{
namespace arm_compute
{

bool isGpuMode()
{
  char *neon = std::getenv("NEON");
  if (neon == nullptr)
    return true;
  else if (neon[0] == '1')
    return false;
  return true;
}

} // namespace arm_compute
} // namespace internal
