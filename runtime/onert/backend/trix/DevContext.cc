/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DevContext.h"

#include <backend/BackendContext.h>
#include <stdexcept>

namespace onert
{
namespace backend
{
namespace trix
{

DevContext::DevContext(int32_t batch_num)
{
  if (batch_num < -1)
  {
    throw std::runtime_error("Invalid batch number" + batch_num);
  }

  auto device_count = getnumNPUdeviceByType(NPUCOND_TRIV2_CONN_SOCIP);
  if (device_count <= 0)
  {
    throw std::runtime_error("Unable to find TRIX NPU device");
  }

  int32_t device_num = batch_num == ContextData::no_batch_parallel ? 0 : batch_num % device_count;

  if (getNPUdeviceByType(&_dev_handle, NPUCOND_TRIV2_CONN_SOCIP, device_num) < 0)
  {
    throw std::runtime_error("Failed to get TRIX NPU device handle");
  }
}

} // namespace trix
} // namespace backend
} // namespace onert
