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

namespace npud
{
namespace backend
{
namespace trix
{

using namespace ::npud::core;

using Handle = void *;

struct TrixDevice
{
  std::vector<Handle> handles;
};

class TrixBackend : public Backend
{
public:
  TrixBackend();
  ~TrixBackend();

  NpuStatus getVersion(std::string &version) override;
  NpuStatus createContext(int device_fd, int priority, NpuContext **ctx) override;
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
