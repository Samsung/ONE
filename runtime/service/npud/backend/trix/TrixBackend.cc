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

NpudStatus TrixBackend::getVersion(std::string &version)
{
  VERBOSE(TrixBackend) << "getVersion" << std::endl;
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::createContext(NpudDevice *device, int device_fd, int priority,
                                      NpudContext *ctx)
{
  VERBOSE(TrixBackend) << "createContext" << std::endl;
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::destroyContext(NpudDevice *device, NpudContext ctx)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::createBuffer(NpudDevice *device, GenericBuffer *buffer)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::destroyBuffer(NpudDevice *device, GenericBuffer *buffer)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::registerModel(NpudDevice *device, const std::string &modelPath,
                                      ModelID *modelId)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::unregisterModel(NpudDevice *device, ModelID modelId)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::createRequest(NpudDevice *device, ModelID modelId, RequestID *requestId)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::destroyRequest(NpudDevice *device, RequestID requestId)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::setRequestData(NpudDevice *device, RequestID requestId,
                                       InputBuffers *input_bufs, TensorDataInfo *in_info,
                                       OutputBuffers *output_bufs, TensorDataInfo *out_info)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

NpudStatus TrixBackend::submitRequest(NpudDevice *device, RequestID requestId)
{
  return NPUD_STATUS_NOT_SUPPORTED;
}

} // namespace trix
} // namespace backend
} // namespace npud
