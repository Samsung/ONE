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

#ifndef __ONERT_BACKEND_TRIX_DEV_CONTEXT_H__
#define __ONERT_BACKEND_TRIX_DEV_CONTEXT_H__

#include <libnpuhost.h>

namespace onert
{
namespace backend
{
namespace trix
{

class DevContext
{
public:
  DevContext()
  {
    auto device_count = getnumNPUdeviceByType(NPUCOND_TRIV2_CONN_SOCIP);
    if (device_count <= 0)
    {
      throw std::runtime_error("Unable to find TRIV2 NPU device");
    }

    // Use NPU 0 device
    if (getNPUdeviceByType(&_dev_handle, NPUCOND_TRIV2_CONN_SOCIP, 0) < 0)
    {
      throw std::runtime_error("Failed to get TRIV2 NPU device handle");
    }
  }

  ~DevContext()
  {
    if (_dev_handle != nullptr)
    {
      unregisterNPUmodel_all(_dev_handle);
      putNPUdevice(_dev_handle);
    }
  }

  npudev_h getDev() { return _dev_handle; }

private:
  // NPU device handle
  // TODO Support multicore npu device
  npudev_h _dev_handle;
};

} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_DEV_CONTEXT_H__
