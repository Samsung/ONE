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

#include "TrixBackend.h"

#include "util/Logging.h"

#if defined(__linux__)
extern "C" {
using namespace ::npud::backend::trix;

TrixBackend *allocate() { return new TrixBackend(); }

void deallocate(TrixBackend *trix) { delete trix; }
}
#endif

namespace npud
{
namespace backend
{
namespace trix
{

TrixBackend::TrixBackend() : _dev_type(NPUCOND_TRIV2_CONN_SOCIP)
{

}

NpuStatus TrixBackend::getVersion(std::string &version)
{
  VERBOSE(TrixBackend) << "getVersion" << std::endl;
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::createContext(NpuDevice *device, int device_fd, int priority,
                                      NpuContext **ctx)
{
  VERBOSE(TrixBackend) << __FUNCTION__ << std::endl;

  auto coreNum = getnumNPUdeviceByType(_dev_type);
  if (coreNum <= 0) {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  std::vector<npudev_h> _handles;
  for (int i = 0; i < coreNum; ++i) {
    npudev_h handle;
    if (getNPUdeviceByType(&handle, _dev_type, i) < 0) {
      // Note Run for all cores.
      continue;
    }
    _handles.emplace_back(handle);
  }

  if (_handles.size() == 0) {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  NpuContext *context = new NpuContext();
  context->_handles = std::move(_handles);
  *ctx = context;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyContext(NpuDevice *device, NpuContext ctx)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::createBuffer(NpuDevice *device, GenericBuffer *buffer)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::destroyBuffer(NpuDevice *device, GenericBuffer *buffer)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::registerModel(NpuDevice *device, const std::string &modelPath,
                                      ModelID *modelId)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::unregisterModel(NpuDevice *device, ModelID modelId)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::createRequest(NpuDevice *device, ModelID modelId, RequestID *requestId)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::destroyRequest(NpuDevice *device, RequestID requestId)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::setRequestData(NpuDevice *device, RequestID requestId,
                                       InputBuffers *input_bufs, TensorDataInfo *in_info,
                                       OutputBuffers *output_bufs, TensorDataInfo *out_info)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::submitRequest(NpuDevice *device, RequestID requestId)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

} // namespace trix
} // namespace backend
} // namespace npud
