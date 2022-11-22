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

TrixBackend::TrixBackend() : _devType(NPUCOND_TRIV2_CONN_SOCIP)
{
  auto coreNum = getnumNPUdeviceByType(_devType);
  if (coreNum <= 0)
  {
    return;
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
    return;
  }

  _dev = std::make_unique<TrixDevice>();
  _dev->handles = std::move(handles);
}

TrixBackend::~TrixBackend()
{
  for (const auto &handle : _dev->handles)
  {
    unregisterNPUmodel_all(handle);
    putNPUdevice(handle);
  }
}

NpuStatus TrixBackend::getVersion(std::string &version)
{
  // TODO Implement details
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::createContext(int deviceId, int priority, NpuContext **ctx)
{
  NpuContext *context = new NpuContext();
  if (deviceId >= _dev->handles.size())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }
  context->defaultCore = deviceId;
  // TODO Consider priority.
  *ctx = context;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyContext(NpuContext *ctx)
{
  if (ctx == nullptr)
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  delete ctx;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::createBuffer(NpuContext *ctx, GenericBuffer *buffer)
{
  // TODO Implement details
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::destroyBuffer(NpuContext *ctx, GenericBuffer *buffer)
{
  // TODO Implement details
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::registerModel(NpuContext *ctx, const std::string &modelPath,
                                     ModelID *modelId)
{
  if (ctx == nullptr)
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ModelID id;
  auto iter = std::find_if(
    _dev->models.begin(), _dev->models.end(), [&](const std::weak_ptr<NpuModelInfo> &p) {
      auto info = p.lock();
      return info ? (info->core == ctx->defaultCore && info->path == modelPath) : false;
    });
  // Already registered model.
  if (iter != _dev->models.end())
  {
    auto info = iter->lock();
    id = info->id;

    auto mIter = ctx->models.find(id);
    if (mIter == ctx->models.end())
    {
      ctx->models.insert(std::make_pair(id, info));
    }
  }
  else
  {
    auto meta = getNPUmodel_metadata(modelPath.c_str(), false);
    if (meta == nullptr)
    {
      return NPU_STATUS_ERROR_OPERATION_FAILED;
    }

    generic_buffer fileInfo;
    fileInfo.type = BUFFER_FILE;
    fileInfo.filepath = modelPath.c_str();
    fileInfo.size = meta->size;

    npudev_h handle = _dev->handles.at(ctx->defaultCore);
    if (registerNPUmodel(handle, &fileInfo, &id) < 0)
    {
      return NPU_STATUS_ERROR_OPERATION_FAILED;
    }

    auto info = std::shared_ptr<NpuModelInfo>(new NpuModelInfo{id, modelPath, ctx->defaultCore});
    ctx->models.insert(std::make_pair(id, info));
    _dev->models.emplace_back(info);
  }

  *modelId = id;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::unregisterModel(NpuContext *ctx, ModelID modelId)
{
  if (ctx == nullptr)
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto iter = ctx->models.find(modelId);
  if (iter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  ctx->models.erase(iter);

  auto mIter = std::remove_if(_dev->models.begin(), _dev->models.end(),
                              [&](const std::weak_ptr<NpuModelInfo> &p) { return p.expired(); });
  _dev->models.erase(mIter, _dev->models.end());
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::createRequest(NpuContext *ctx, ModelID modelId, RequestID *requestId)
{
  if (ctx == nullptr)
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto iter = ctx->models.find(modelId);
  if (iter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  int id;
  npudev_h handle = _dev->handles.at(iter->second->core);
  if (createNPU_request(handle, modelId, &id) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  auto &requestMap = ctx->requests;
  requestMap.insert({id, std::unique_ptr<NpuRequestInfo>(new NpuRequestInfo{id, modelId})});

  *requestId = id;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyRequest(NpuContext *ctx, RequestID requestId)
{
  if (ctx == nullptr)
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto &requestMap = ctx->requests;
  auto iter = requestMap.find(requestId);
  if (iter == requestMap.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ModelID modelId = iter->second->modelId;
  auto miter = ctx->models.find(modelId);
  if (miter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  npudev_h handle = _dev->handles.at(miter->second->core);
  if (removeNPU_request(handle, requestId) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  requestMap.erase(iter);
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::setRequestData(NpuContext *ctx, RequestID requestId, InputBuffers *inputBufs,
                                      TensorDataInfos *inputInfos, OutputBuffers *outputBufs,
                                      TensorDataInfos *outputInfos)
{
  // TODO Implement details
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::submitRequest(NpuContext *ctx, RequestID requestId)
{
  // TODO Implement details
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

} // namespace trix
} // namespace backend
} // namespace npud
