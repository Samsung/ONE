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

#ifndef __ONE_SERVICE_NPUD_CORE_BACKEND_H__
#define __ONE_SERVICE_NPUD_CORE_BACKEND_H__

#include "ir/Layout.h"
#include "ir/DataType.h"

#include <string>
#include <vector>
#include <map>

namespace npud
{
namespace core
{

/* Backend supports up to 16 buffers. */
#define NPU_TENSOR_CARDINALITY_MAX (16)

using Handle = void *;
/**
 * @brief Npu model ID
 *
 */
using ModelID = uint32_t;

/**
 * @brief Npu request ID
 *
 */
using RequestID = uint32_t;

/**
 * @brief Various kinds of buffer supported for input/output/model.
 *
 */
using GenericBuffer = void *;
struct GenericBuffers
{
  uint32_t numBuffers;
  GenericBuffer buffers[NPU_TENSOR_CARDINALITY_MAX];
};

/**
 * @brief Npu input/output buffers are compotible with GenericBuffers.
 *
 */
typedef GenericBuffers InputBuffers;
typedef GenericBuffers OutputBuffers;

/**
 * @brief Npu tensor data info description.
 *
 */
struct TensorDataInfo
{
  ir::Layout layout;
  ir::DataType type;
};

/**
 * @brief Npu error status.
 *
 */
enum NpuStatus
{
  NPU_STATUS_SUCCESS = 0,
  NPU_STATUS_ERROR_OPERATION_FAILED,
  NPU_STATUS_ERROR_NOT_SUPPORTED,
  NPU_STATUS_ERROR_INVALID_ARGUMENT,
};

/**
 * @brief Npu device definition
 *
 */
struct NpuDevice
{
  int fd;
};

/**
 * @brief Npu context definition
 *
 */
struct NpuContext
{
  std::vector<Handle> handles;
  // Note Manage model id per model path.
  //      Do we need to handle the case of requesting different model files
  //      with the same model path?
  std::vector<std::map<ModelID, const std::string>> modelIds;
  std::map<RequestID, ModelID> requestIds;
  int defaultCore;
};

/**
 * @brief Npu backend interface
 *
 * @detail Backend module should implement this Backend interface.
 *         Npu daemon will load this class symbol at runtime.
 */
class Backend
{
public:
  virtual ~Backend() = default;

  virtual NpuStatus getVersion(std::string &version) = 0;
  virtual NpuStatus createContext(NpuDevice *device, int device_fd, int priority,
                                  NpuContext **ctx) = 0;
  virtual NpuStatus destroyContext(NpuDevice *device, NpuContext *ctx) = 0;
  virtual NpuStatus createBuffer(NpuDevice *device, GenericBuffer *buffer) = 0;
  virtual NpuStatus destroyBuffer(NpuDevice *device, GenericBuffer *buffer) = 0;
  // TODO Support to register model from buffer
  virtual NpuStatus registerModel(NpuDevice *device, NpuContext *ctx, const std::string &modelPath,
                                  ModelID *modelId) = 0;
  virtual NpuStatus unregisterModel(NpuDevice *device, NpuContext *ctx, ModelID modelId) = 0;
  virtual NpuStatus createRequest(NpuDevice *device, NpuContext *ctx, ModelID modelId,
                                  RequestID *requestId) = 0;
  virtual NpuStatus destroyRequest(NpuDevice *device, RequestID requestId) = 0;
  virtual NpuStatus setRequestData(NpuDevice *device, RequestID requestId, InputBuffers *input_bufs,
                                   TensorDataInfo *in_info, OutputBuffers *output_bufs,
                                   TensorDataInfo *out_info) = 0;
  virtual NpuStatus submitRequest(NpuDevice *device, RequestID requestId) = 0;
};

// std::string allocateSymbol("allocate");
// std::string deallocateSymbol("deallocate");

typedef Backend *(*NpuAlloc)();
typedef void (*NpuDealloc)(Backend *);

} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_BACKEND_H__
