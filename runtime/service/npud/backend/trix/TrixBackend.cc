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

data_type convertDataType(const ir::DataType type)
{
  switch (type)
  {
    case ir::DataType::QUANT_UINT8_ASYMM:
      return DATA_TYPE_QASYMM8;
    case ir::DataType::QUANT_INT16_SYMM:
      return DATA_TYPE_QSYMM16;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

data_layout convertDataLayout(const ir::Layout layout)
{
  switch (layout)
  {
    case ir::Layout::NCHW:
      return DATA_LAYOUT_NCHW;
    case ir::Layout::NHWC:
      return DATA_LAYOUT_NHWC;
    default:
      throw std::runtime_error("Unknown Layout");
  }
}

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
  for (auto &ctx : _dev->ctxs)
  {
    // TODO Exception controll of `at`
    npudev_h handle = _dev->handles.at(ctx->defaultCore);
    for (auto &p : ctx->requests)
    {
      removeNPU_request(handle, p.first);
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
  VERBOSE(TrixBackend) << "getVersion" << std::endl;
  return NPU_STATUS_ERROR_NOT_SUPPORTED;
}

NpuStatus TrixBackend::createContext(int device_fd, int priority, NpuContext **ctx)
{
  auto context = std::make_unique<NpuContext>();
  context->defaultCore = device_fd;
  *ctx = context.get();
  _dev->ctxs.emplace_back(std::move(context));
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyContext(NpuContext *ctx)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // TODO Remove request items
  for (auto &p : ctx->requests)
  {
    // TODO Exception controll of `at`
    npudev_h handle = _dev->handles.at(ctx->defaultCore);
    if (removeNPU_request(handle, p.first) < 0)
    {
      return NPU_STATUS_ERROR_OPERATION_FAILED;
    }
  }

  for (auto id : ctx->models)
  {
    auto &info = _dev->models.at(id);
    if (info->refCount - 1 == 0)
    {
      npudev_h handle = _dev->handles.at(ctx->defaultCore);
      if (unregisterNPUmodel(handle, id) < 0)
      {
        return NPU_STATUS_ERROR_OPERATION_FAILED;
      }
    }
    if (--info->refCount == 0)
    {
      _dev->models.erase(id);
    }
    // p.second.reset();
  }
  // ctx->models.clear();

  _dev->ctxs.erase(citer);
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::createBuffers(NpuContext *ctx, GenericBuffers *bufs)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  generic_buffers *buffers = reinterpret_cast<generic_buffers *>(bufs);
  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (allocNPU_genericBuffers(handle, buffers) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyBuffers(NpuContext *ctx, GenericBuffers *bufs)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  generic_buffers *buffers = reinterpret_cast<generic_buffers *>(bufs);
  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (cleanNPU_genericBuffers(handle, buffers) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::registerModel(NpuContext *ctx, const std::string &modelPath,
                                     ModelID *modelId)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ModelID id;
  auto iter =
    std::find_if(_dev->models.begin(), _dev->models.end(),
                 [&](const std::pair<const ModelID, std::unique_ptr<TrixModelInfo>> &p) {
                   return p.second->core == ctx->defaultCore && p.second->path == modelPath;
                 });
  // auto iter = std::find_if(_dev->models.begin(), _dev->models.end(),
  //                          [&](const std::weak_ptr<NpuModelInfo> &p) {
  //                            auto info = p.lock();
  //                            if (info)
  //                            {
  //                              return info->core == ctx->defaultCore && info->path == modelPath;
  //                            }
  //                            else
  //                            {
  //                              return false;
  //                            }
  //                          });
  // Already registered model.
  if (iter != _dev->models.end())
  {
    ctx->models.emplace_back(iter->first);
    _dev->models.at(iter->first)->refCount++;
    // auto info = iter->lock();
    // id = info->id;

    // auto mIter = ctx->models.find(id);
    // if (mIter == ctx->models.end())
    // {
    //   ctx->models.insert(std::make_pair(id, info));
    // }
  }
  else
  {
    auto meta = getNPUmodel_metadata(modelPath.c_str(), false);
    if (meta == nullptr)
    {
      return NPU_STATUS_ERROR_OPERATION_FAILED;
    }

    if (NPUBIN_VERSION(meta->magiccode) != 3)
    {
      return NPU_STATUS_ERROR_INVALID_MODEL;
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

    ctx->models.emplace_back(id);
    // auto info = std::make_unique<TrixModelInfo>(id, modelPath, ctx->defaultCore, meta);
    _dev->models.insert(std::make_pair(id, std::unique_ptr<TrixModelInfo>(new TrixModelInfo(
                                             id, modelPath, ctx->defaultCore, meta))));
    _dev->models.at(id)->refCount++;
  }

  *modelId = id;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::unregisterModel(NpuContext *ctx, ModelID modelId)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto iter = std::find(ctx->models.begin(), ctx->models.end(), modelId);
  // auto iter = ctx->models.find(modelId);
  if (iter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  auto &info = _dev->models.at(modelId);
  if (info->refCount - 1 == 0)
  {
    npudev_h handle = _dev->handles.at(ctx->defaultCore);
    if (unregisterNPUmodel(handle, modelId) < 0)
    {
      return NPU_STATUS_ERROR_OPERATION_FAILED;
    }
  }

  ctx->models.erase(iter);
  if (--info->refCount == 0)
  {
    _dev->models.erase(modelId);
  }
  // iter->second.reset();
  // ctx->models.erase(iter);

  // auto mIter = std::remove_if(_dev->models.begin(), _dev->models.end(),
  //                             [&](const std::weak_ptr<NpuModelInfo> &p) { return p.expired(); });
  // _dev->models.erase(mIter, _dev->models.end());
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::createRequest(NpuContext *ctx, ModelID modelId, RequestID *requestId)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto iter = std::find(ctx->models.begin(), ctx->models.end(), modelId);
  if (iter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  int id;
  // TODO Exception controll of `at`
  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (createNPU_request(handle, modelId, &id) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  auto &requestMap = ctx->requests;
  requestMap.insert(
    {id, std::unique_ptr<NpuRequestInfo>(new NpuRequestInfo(id, modelId, ctx->defaultCore))});

  *requestId = id;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::destroyRequest(NpuContext *ctx, RequestID requestId)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto &requestMap = ctx->requests;
  auto iter = requestMap.find(requestId);
  if (iter == requestMap.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // TODO Exception controll of `at`
  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (removeNPU_request(handle, requestId) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  requestMap.erase(iter);
  return NPU_STATUS_SUCCESS;
}

// void TrixBackend::setDataInfo(GenericBuffers *bufs, tensor_data_info *info)
// {
//   info->num_info = bufs->numBuffers;

//   for (int i = 0; i < info->num_info; ++i) {
//     info->info[i].layout =
//   }
// }

// void TrixBackend::setBuffers(GenericBuffers *bufs, genereic_buffers *bufs)
// {

// }

NpuStatus TrixBackend::setRequestData(NpuContext *ctx, RequestID requestId, InputBuffers *inputBufs,
                                      OutputBuffers *outputBufs)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto &requestMap = ctx->requests;
  auto iter = requestMap.find(requestId);
  if (iter == requestMap.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ModelID modelId = iter->second->modelId;
  auto miter = std::find(ctx->models.begin(), ctx->models.end(), modelId);
  if (miter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_DATA;
  }

  // TODO Exception controll of `at`
  auto &minfo = _dev->models.at(modelId);
  if (minfo->meta->input_seg_num != inputBufs->numBuffers ||
      minfo->meta->output_seg_num != outputBufs->numBuffers)
  {
    return NPU_STATUS_ERROR_INVALID_DATA;
  }

  tensors_data_info inInfos;
  tensors_data_info outInfos;

  inInfos.num_info = inputBufs->numBuffers;
  for (auto i = 0; i < inInfos.num_info; ++i)
  {
    inInfos.info[i].layout = DATA_LAYOUT_MODEL;
    inInfos.info[i].type = minfo->meta->input_seg_quant_type[i];
  }

  outInfos.num_info = outputBufs->numBuffers;
  for (auto i = 0; i < outInfos.num_info; ++i)
  {
    outInfos.info[i].layout = DATA_LAYOUT_MODEL;
    outInfos.info[i].type = minfo->meta->output_seg_quant_type[i];
  }

  // setDataInfo(inputBufs, &inInfos);
  // setDataInfo(outputBufs, &outInfos);

  // setBuffers(inputBufs, &inBufs);
  // setBuffers(outputBufs, &outBufs);

  // inInfos.num_info = inputInfos->numInfos;
  // for (int i = 0; i < inputInfos->numInfos; ++i)
  // {
  //   inInfos.info[i].layout = convertDataLayout(inputInfos->infos[i].layout);
  //   inInfos.info[i].type = convertDataType(inputInfos->infos[i].type);
  // }

  // outInfos.num_info = outputInfos->numInfos;
  // for (int i = 0; i < outputInfos->numInfos; ++i)
  // {
  //   outInfos.info[i].layout = convertDataLayout(outputInfos->infos[i].layout);
  //   outInfos.info[i].type = convertDataType(outputInfos->infos[i].type);
  // }

  input_buffers inBufs;
  output_buffers outBufs;

  inBufs.num_buffers = inputBufs->numBuffers;
  for (auto i = 0; i < inBufs.num_buffers; ++i)
  {
    inBufs.bufs[i].addr = inputBufs->buffers[i].addr;
    inBufs.bufs[i].size = inputBufs->buffers[i].size;
    inBufs.bufs[i].type = BUFFER_MAPPED;
  }

  outBufs.num_buffers = outputBufs->numBuffers;
  for (auto i = 0; i < outBufs.num_buffers; ++i)
  {
    outBufs.bufs[i].addr = outputBufs->buffers[i].addr;
    outBufs.bufs[i].size = outputBufs->buffers[i].size;
    outBufs.bufs[i].type = BUFFER_MAPPED;
  }

  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (setNPU_requestData(handle, requestId, &inBufs, &inInfos, &outBufs, &outInfos) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  // iter->second->inBufs = inputBufs;
  // iter->second->inInfos = inputInfos;
  // iter->second->outBufs = outputBufs;
  // iter->second->outInfos = outputInfos;
  return NPU_STATUS_SUCCESS;
}

NpuStatus TrixBackend::submitRequest(NpuContext *ctx, RequestID requestId)
{
  auto citer = std::find_if(_dev->ctxs.begin(), _dev->ctxs.end(), [&](std::unique_ptr<NpuContext> &c) { return c.get() == ctx; });
  if (citer == _dev->ctxs.end()) {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto &requestMap = ctx->requests;
  auto iter = requestMap.find(requestId);
  if (iter == requestMap.end())
  {
    return NPU_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ModelID modelId = iter->second->modelId;
  auto miter = std::find(ctx->models.begin(), ctx->models.end(), modelId);
  if (miter == ctx->models.end())
  {
    return NPU_STATUS_ERROR_INVALID_MODEL;
  }

  npudev_h handle = _dev->handles.at(ctx->defaultCore);
  if (submitNPU_request(handle, requestId) < 0)
  {
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }

  return NPU_STATUS_SUCCESS;
}

} // namespace trix
} // namespace backend
} // namespace npud
