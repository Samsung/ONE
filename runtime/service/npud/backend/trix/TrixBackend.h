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
// #include <libnpuhost.h>
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
  TrixBackend() = default;
  ~TrixBackend() = default;

  NpudStatus getVersion(std::string &version) override;
  NpudStatus createContext(NpudDevice *device, int device_fd, int priority,
                           NpudContext *ctx) override;
  NpudStatus destroyContext(NpudDevice *device, NpudContext ctx) override;
  NpudStatus createBuffer(NpudDevice *device, GenericBuffer *buffer) override;
  NpudStatus destroyBuffer(NpudDevice *device, GenericBuffer *buffer) override;
  // TODO Support to register model from buffer
  NpudStatus registerModel(NpudDevice *device, const std::string &modelPath,
                           ModelID *modelId) override;
  NpudStatus unregisterModel(NpudDevice *device, ModelID modelId) override;
  NpudStatus createRequest(NpudDevice *device, ModelID modelId, RequestID *requestId) override;
  NpudStatus destroyRequest(NpudDevice *device, RequestID requestId) override;
  NpudStatus setRequestData(NpudDevice *device, RequestID requestId, InputBuffers *input_bufs,
                            TensorDataInfo *in_info, OutputBuffers *output_bufs,
                            TensorDataInfo *out_info) override;
  NpudStatus submitRequest(NpudDevice *device, RequestID requestId) override;
};

} // namespace trix
} // namespace backend
} // namespace npud

#endif // __ONE_SERVICE_NPUD_BACKEND_TRIX_BACKEND_H__
