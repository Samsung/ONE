/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ICLTensor.h"

#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/core/CL/OpenCL.h>

namespace onert
{
namespace backend
{
namespace acl_cl
{
namespace operand
{

void ICLTensor::access(const std::function<void(ITensor &tensor)> &fn)
{
  auto &queue = ::arm_compute::CLScheduler::get().queue();

  // This is an optional input
  if (total_size() == 0)
    return;

  map(queue);
  fn(*this);
  unmap(queue);
}

void ICLTensor::enqueueWriteBuffer(const void *ptr, bool blocking)
{
  auto &queue = ::arm_compute::CLScheduler::get().queue();
  queue.enqueueWriteBuffer(handle()->cl_buffer(), blocking ? CL_TRUE : CL_FALSE, 0,
                           info()->total_size(), ptr);
}

void ICLTensor::enqueueReadBuffer(void *ptr, bool blocking)
{
  auto &queue = ::arm_compute::CLScheduler::get().queue();
  queue.enqueueReadBuffer(handle()->cl_buffer(), blocking ? CL_TRUE : CL_FALSE, 0,
                          info()->total_size(), ptr);
}
} // namespace operand
} // namespace acl_cl
} // namespace backend
} // namespace onert
