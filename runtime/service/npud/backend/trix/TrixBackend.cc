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

TrixBackend::TrixBacked(): _dev_type(NPUCOND_TRIV2_CONN_SOCIP)
{

}

NpudStatus TrixBackend::getVersion(std::string &version)
{
  VERBOSE(TrixBackend) << "getVersion" << std::endl;
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::createContext(NpudDevice *device, int device_fd, int priority,
                                      NpudContext **ctx)
{
  VERBOSE(TrixBackend) << __FUNCTION__ << std::endl;

  auto coreNum = getnumNPUdeviceByType(_dev_type);
  if (coreNum <= 0) {
    return NPUD_STATUS_ERROR_OPERATION_FAILED;
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

  if (_handle.size() == 0) {
    return NPUD_STATUS_ERROR_OPERATION_FAILED;
  }

  NpudContext *context = new NpudContext();
  context->_handles = std::move(_handles);
  return NPUD_STATUS_SUCCESS;
}

NpudStatus TrixBackend::destroyContext(NpudDevice *device, NpudContext ctx)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::createBuffer(NpudDevice *device, GenericBuffer *buffer)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::destroyBuffer(NpudDevice *device, GenericBuffer *buffer)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::registerModel(NpudDevice *device, const std::string &modelPath,
                                      ModelID *modelId)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::unregisterModel(NpudDevice *device, ModelID modelId)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::createRequest(NpudDevice *device, ModelID modelId, RequestID *requestId)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::destroyRequest(NpudDevice *device, RequestID requestId)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::setRequestData(NpudDevice *device, RequestID requestId,
                                       InputBuffers *input_bufs, TensorDataInfo *in_info,
                                       OutputBuffers *output_bufs, TensorDataInfo *out_info)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

NpudStatus TrixBackend::submitRequest(NpudDevice *device, RequestID requestId)
{
  return NPUD_STATUS_ERROR_NOT_SUPPORTED;
}

} // namespace trix
} // namespace backend
} // namespace npud
