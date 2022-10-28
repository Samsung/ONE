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
#include <algorithm>

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

TrixBackend::TrixBackend() : _devType(NPUCOND_TRIV2_CONN_SOCIP) {}

NpuStatus TrixBackend::getVersion(std::string &version)
{
  VERBOSE(TrixBackend) << "getVersion" << std::endl;
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::createContext(NpuDevice *device, int device_fd, int priority,
                                     NpuContext **ctx)
{
  VERBOSE(TrixBackend) << __FUNCTION__ << std::endl;

  auto coreNum = getnumNPUdeviceByType(_devType);
  if (coreNum <= 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  std::vector<npudev_h> handles;
  for (int i = 0; i < coreNum; ++i)
  {
    npudev_h handle;
    if (getNPUdeviceByType(&handle, _devType, i) < 0)
    {
      // Note Run for all cores.
      continue;
    }
    handles.emplace_back(handle);
  }

  if (handles.size() == 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  NpuContext *context = new NpuContext();
  context->handles = std::move(handles);
  context->modelIds.resize(context->handles.size());
  context->defaultCore = 0;
  *ctx = context;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyContext(NpuDevice *device, NpuContext *ctx)
{
  VERBOSE(TrixBackend) << __FUNCTION__ << std::endl;

  for (const auto &handle : ctx->handles)
  {
    unregisterNPUmodel_all(handle);
    putNPUdevice(handle);
  }
  delete ctx;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::createBuffer(NpuDevice *device, GenericBuffer *buffer)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::destroyBuffer(NpuDevice *device, GenericBuffer *buffer)
{
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::registerModel(NpuDevice *device, NpuContext *ctx,
                                     const std::string &modelPath, ModelID *modelId)
{
  auto &modelMap = ctx->modelIds.at(ctx->defaultCore);
  auto iter = std::find_if(
    modelMap.begin(), modelMap.end(),
    [&](const std::pair<ModelID, const std::string> &p) { return p.second == modelPath; });
  if (iter != modelMap.end())
  {
    *modelId = iter->first;
    return NPU_STATUS_SUCCESS;
  }

  auto meta = getNPUmodel_metadata(modelPath.c_str(), false);
  if (meta == nullptr)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  generic_buffer fileInfo;
  fileInfo.type = BUFFER_FILE;
  fileInfo.filepath = modelPath.c_str();
  fileInfo.size = meta->size;

  ModelID id;
  npudev_h handle = ctx->handles.at(ctx->defaultCore);
  if (registerNPUmodel(handle, &fileInfo, &id) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  modelMap.insert({id, modelPath});
  *modelId = id;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::unregisterModel(NpuDevice *device, NpuContext *ctx, ModelID modelId)
{
  // Note Unregister model from the same default core's handle.
  auto &modelMap = ctx->modelIds.at(ctx->defaultCore);
  auto iter = modelMap.find(modelId);
  if (iter == modelMap.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  npudev_h handle = ctx->handles.at(ctx->defaultCore);
  if (unregisterNPUmodel(handle, modelId) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  modelMap.erase(iter);
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::createRequest(NpuDevice *device, NpuContext *ctx, ModelID modelId,
                                     RequestID *requestId)
{
  auto &modelMap = ctx->modelIds.at(ctx->defaultCore);
  auto iter = modelMap.find(modelId);
  if (iter == modelMap.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  int id;
  npudev_h handle = ctx->handles.at(ctx->defaultCore);
  if (createNPU_request(handle, modelId, &id) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  auto &requestMap = ctx->requestIds;
  requestMap.insert({id, modelId});
  *requestId = id;
  return NPU_STATUS_SUCCESS;
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
