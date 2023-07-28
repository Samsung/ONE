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
  for (const auto &ctx : _dev->ctxs)
  {
    npudev_h handle = _dev->handles.at(ctx->defaultCore);
    for (const auto id : ctx->requests)
    {
      removeNPU_request(handle, id);
    }
  }

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
  if (deviceId >= _dev->handles.size())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }
  auto context = std::make_unique<NpuContext>();
  context->defaultCore = deviceId;
  // TODO Consider priority.
  *ctx = context.get();
  _dev->ctxs.emplace_back(std::move(context));
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyContext(NpuContext *ctx)
{
  if (ctx == nullptr)
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(),
                            [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  npudev_h handle = _dev->handles.at(ctx->defaultCore);

  for (auto &&rid : ctx->requests)
  {
    if (removeNPU_request(handle, rid) < 0)
    {
      return NPU_STATUS_ERROR_OPERATION_FAILED;
    }
    _dev->requests.erase(rid);
  }

  for (auto &&mid : ctx->models)
  {
    auto &minfo = _dev->models.at(mid);
    if (--minfo->refCount == 0)
    {
      if (unregisterNPUmodel(handle, mid) < 0)
      {
        return NPU_STATUS_ERROR_OPERATION_FAILED;
      }
      _dev->models.erase(mid);
    }
  }

  _dev->ctxs.erase(citer);
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

  ModelID id = 0;
  auto iter =
    std::find_if(_dev->models.begin(), _dev->models.end(),
                 [&](const std::pair<const ModelID, std::unique_ptr<TrixModelInfo>> &p) {
                   return p.second->core == ctx->defaultCore && p.second->path == modelPath;
                 });
  // Already registered model.
  if (iter != _dev->models.end())
  {
    _dev->models.at(iter->first)->refCount++;
    ctx->models.emplace_back(iter->first);
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

    _dev->models.insert(std::make_pair(id, std::unique_ptr<TrixModelInfo>(new TrixModelInfo{
                                             id, modelPath, ctx->defaultCore, meta, 1})));
    ctx->models.emplace_back(id);
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

  auto miter = std::find(ctx->models.begin(), ctx->models.end(), modelId);
  if (miter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  npudev_h handle = _dev->handles.at(ctx->defaultCore);

  for (auto riter = ctx->requests.begin(); riter != ctx->requests.end();)
  {
    auto &rinfo = _dev->requests.at(*riter);
    if (rinfo->modelId == modelId)
    {
      if (removeNPU_request(handle, rinfo->id) < 0)
      {
        return NPU_STATUS_ERROR_OPERATION_FAILED;
      }
      _dev->requests.erase(rinfo->id);
      riter = ctx->requests.erase(riter);
    }
    else
    {
      ++riter;
    }
  }

  auto &minfo = _dev->models.at(modelId);
  if (--minfo->refCount == 0)
  {
    if (unregisterNPUmodel(handle, modelId) < 0)
    {
      return NPU_STATUS_ERROR_OPERATION_FAILED;
    }
    _dev->models.erase(modelId);
  }

  ctx->models.erase(miter);
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::createRequest(NpuContext *ctx, ModelID modelId, RequestID *requestId)
{
  if (ctx == nullptr)
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto miter = std::find(ctx->models.begin(), ctx->models.end(), modelId);
  if (miter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  int id = 0;
  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (createNPU_request(handle, modelId, &id) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  _dev->requests.insert(std::make_pair(id, std::unique_ptr<TrixRequestInfo>(new TrixRequestInfo{
                                             static_cast<RequestID>(id), modelId})));
  ctx->requests.emplace_back(id);

  *requestId = id;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyRequest(NpuContext *ctx, RequestID requestId)
{
  if (ctx == nullptr)
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto riter = std::find(ctx->requests.begin(), ctx->requests.end(), requestId);
  if (riter == ctx->requests.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (removeNPU_request(handle, requestId) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  _dev->requests.erase(requestId);
  ctx->requests.erase(riter);
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::setRequestData(NpuContext *ctx, RequestID requestId, InputBuffers *inputBufs,
                                      TensorDataInfos *inputInfos, OutputBuffers *outputBufs,
                                      TensorDataInfos *outputInfos)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(),
                            [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto riter = std::find(ctx->requests.begin(), ctx->requests.end(), requestId);
  if (riter == ctx->requests.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto &req = _dev->requests.at(requestId);
  auto miter = std::find(ctx->models.begin(), ctx->models.end(), req->modelId);
  if (miter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  // TODO Exception controll of `at`
  auto &minfo = _dev->models.at(req->modelId);
  if (minfo->meta->input_seg_num != inputBufs->numBuffers ||
      minfo->meta->output_seg_num != outputBufs->numBuffers)
  {
    return NPU_STATUS_ERROR_INVALID_DATA;
  }

  auto &inInfos = req->inInfos;
  auto &outInfos = req->outInfos;

  inInfos->num_info = inputBufs->numBuffers;
  for (auto i = 0; i < inInfos->num_info; ++i)
  {
    inInfos->info[i].layout = DATA_LAYOUT_MODEL;
    inInfos->info[i].type = minfo->meta->input_seg_quant_type[i];
  }

  outInfos->num_info = outputBufs->numBuffers;
  for (auto i = 0; i < outInfos->num_info; ++i)
  {
    outInfos->info[i].layout = DATA_LAYOUT_MODEL;
    outInfos->info[i].type = minfo->meta->output_seg_quant_type[i];
  }

  auto &inBufs = req->inBufs;
  auto &outBufs = req->outBufs;

  inBufs->num_buffers = inputBufs->numBuffers;
  for (auto i = 0; i < inBufs->num_buffers; ++i)
  {
    if (inputBufs->buffers[i].type == NPU_BUFFER_MAPPED)
    {
      inBufs->bufs[i].addr = inputBufs->buffers[i].addr;
    }
    else if (inputBufs->buffers[i].type == NPU_BUFFER_DMABUF)
    {
      // TODO Implement details
      // inBufs.bufs[i].dmabuf = inputBufs->buffers[i].dmabuf;
      // inBufs.bufs[i].offset = inputBufs->buffers[i].offset;
    }
    else
    {
      continue;
    }
    inBufs->bufs[i].size = inputBufs->buffers[i].size;
    inBufs->bufs[i].type = static_cast<buffer_types>(inputBufs->buffers[i].type);
  }

  outBufs->num_buffers = outputBufs->numBuffers;
  for (auto i = 0; i < outBufs->num_buffers; ++i)
  {
    if (outputBufs->buffers[i].type == NPU_BUFFER_MAPPED)
    {
      outBufs->bufs[i].addr = outputBufs->buffers[i].addr;
    }
    else if (outputBufs->buffers[i].type == NPU_BUFFER_DMABUF)
    {
      // TODO Implement details
      // outBufs.bufs[i].dmabuf = outputBufs->buffers[i].dmabuf;
      // outBufs.bufs[i].offset = outputBufs->buffers[i].offset;
    }
    else
    {
      continue;
    }
    outBufs->bufs[i].size = outputBufs->buffers[i].size;
    outBufs->bufs[i].type = static_cast<buffer_types>(outputBufs->buffers[i].type);
  }

  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (setNPU_requestData(handle, requestId, inBufs.get(), inInfos.get(), outBufs.get(),
                         outInfos.get()) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::submitRequest(NpuContext *ctx, RequestID requestId)
{
  // TODO Implement details
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

} // namespace trix
} // namespace backend
} // namespace npud
