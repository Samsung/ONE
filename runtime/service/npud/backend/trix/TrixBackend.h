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

#ifndef __ONE_SERVICE_NPUD_BACKEND_TRIX_BACKEND_H__
#define __ONE_SERVICE_NPUD_BACKEND_TRIX_BACKEND_H__

#include <core/Backend.h>
#include <libnpuhost.h>
#include <memory>
#include <vector>
#include <map>

namespace npud
{
namespace backend
{
namespace trix
{

using namespace ::npud::core;

using Handle = void *;

/**
 * @brief Trix model information.
 *
 * @param id The model identifier.
 * @param path The model path.
 * @param core The core number where the model is registered.
 * @param meta The meta data of model.
 * @param refCount The reference count of model users.
 */
struct TrixModelInfo
{
  ModelID id;
  std::string path;
  int core;
  npubin_meta *meta;
  int refCount;

  TrixModelInfo() : meta(nullptr), refCount(0) {}
  TrixModelInfo(ModelID _id, const std::string &_path, int _core, npubin_meta *_meta, int _refCount)
    : id(_id), path(_path), core(_core), meta(_meta), refCount(_refCount)
  {
  }
  ~TrixModelInfo() { free(meta); }
};

/**
 * @brief Trix request information
 *
 * @param id The request identifier.
 * @param modelId The model id of request.
 */
struct TrixRequestInfo
{
  RequestID id;
  ModelID modelId;
  std::unique_ptr<input_buffers> inBufs;
  std::unique_ptr<tensors_data_info> inInfos;
  std::unique_ptr<output_buffers> outBufs;
  std::unique_ptr<tensors_data_info> outInfos;
  TrixRequestInfo(RequestID _id, ModelID _mid)
    : id(_id), modelId(_mid), inBufs(std::make_unique<input_buffers>()),
      inInfos(std::make_unique<tensors_data_info>()), outBufs(std::make_unique<output_buffers>()),
      outInfos(std::make_unique<tensors_data_info>())
  {
  }
};

/**
 * @brief Trix device information
 *
 * @param handles The device handle list.
 * @param ctxs The NpuContext list.
 * @param models The model map.
 * @param requests The request map.
 */
struct TrixDevice
{
  std::vector<Handle> handles;
  std::vector<std::unique_ptr<NpuContext>> ctxs;
  std::map<ModelID, std::unique_ptr<TrixModelInfo>> models;
  std::map<RequestID, std::unique_ptr<TrixRequestInfo>> requests;
};

class TrixBackend : public Backend
{
public:
  TrixBackend();
  ~TrixBackend();

  NpuStatus getVersion(std::string &version) override;
  NpuStatus createContext(int deviceId, int priority, NpuContext **ctx) override;
  NpuStatus destroyContext(NpuContext *ctx) override;
  NpuStatus createBuffer(NpuContext *ctx, GenericBuffer *buffer) override;
  NpuStatus destroyBuffer(NpuContext *ctx, GenericBuffer *buffer) override;
  // TODO Support to register model from buffer
  NpuStatus registerModel(NpuContext *ctx, const std::string &modelPath, ModelID *modelId) override;
  NpuStatus unregisterModel(NpuContext *ctx, ModelID modelId) override;
  NpuStatus createRequest(NpuContext *ctx, ModelID modelId, RequestID *requestId) override;
  NpuStatus destroyRequest(NpuContext *ctx, RequestID requestId) override;
  NpuStatus setRequestData(NpuContext *ctx, RequestID requestId, InputBuffers *inputBufs,
                           TensorDataInfos *inputInfos, OutputBuffers *outputBufs,
                           TensorDataInfos *outputInfos) override;
  NpuStatus submitRequest(NpuContext *ctx, RequestID requestId) override;

private:
  dev_type _devType;
  std::unique_ptr<TrixDevice> _dev;
};

} // namespace trix
} // namespace backend
} // namespace npud

#endif // __ONE_SERVICE_NPUD_BACKEND_TRIX_BACKEND_H__
