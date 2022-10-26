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
#include <vector>
#include <unordered_map>

namespace npud
{
namespace backend
{
namespace trix
{

using namespace ::npud::core;

class TrixBackend : public Backend
{
public:
  TrixBackend();
  ~TrixBackend() = default;

  NpuStatus getVersion(std::string &version) override;
  NpuStatus createContext(NpuDevice *device, int device_fd, int priority,
                           NpuContext **ctx) override;
  NpuStatus destroyContext(NpuDevice *device, NpuContext ctx) override;
  NpuStatus createBuffer(NpuDevice *device, GenericBuffer *buffer) override;
  NpuStatus destroyBuffer(NpuDevice *device, GenericBuffer *buffer) override;
  // TODO Support to register model from buffer
  NpuStatus registerModel(NpuDevice *device, const std::string &modelPath,
                           ModelID *modelId) override;
  NpuStatus unregisterModel(NpuDevice *device, ModelID modelId) override;
  NpuStatus createRequest(NpuDevice *device, ModelID modelId, RequestID *requestId) override;
  NpuStatus destroyRequest(NpuDevice *device, RequestID requestId) override;
  NpuStatus setRequestData(NpuDevice *device, RequestID requestId, InputBuffers *input_bufs,
                            TensorDataInfo *in_info, OutputBuffers *output_bufs,
                            TensorDataInfo *out_info) override;
  NpuStatus submitRequest(NpuDevice *device, RequestID requestId) override;

private:
  dev_type _dev_type;
};

} // namespace trix
} // namespace backend
} // namespace npud

#endif // __ONE_SERVICE_NPUD_BACKEND_TRIX_BACKEND_H__
