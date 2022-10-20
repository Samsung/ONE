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

namespace npud
{
namespace core
{

/* Backend supports up to 16 buffers. */
#define NPU_TENSOR_CARDINALITY_MAX (16)

/**
 * @brief Npud device definition
 *
 */
struct NpudDevice
{
  int fd;
  std::string name;
};

/**
 * @brief Npud device handle
 *
 */
using NpudContext = uint64_t;

/**
 * @brief Npud model ID
 *
 */
using ModelID = uint32_t;

/**
 * @brief Npud request ID
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
 * @brief Npud input/output buffers are compotible with GenericBuffers.
 *
 */
typedef GenericBuffers InputBuffers;
typedef GenericBuffers OutputBuffers;

/**
 * @brief Npud tensor data info description.
 *
 */
struct TensorDataInfo
{
  ir::Layout layout;
  ir::DataType type;
};

/**
 * @brief Npud error status.
 *
 */
enum NpudStatus
{
  NPUD_STATUS_SUCCESS = 0,
  NPUD_STATUS_ERROR_PERM = 1,
  NPUD_STATUS_NOT_SUPPORTED = 2,
};

/**
 * @brief Npud backend interface
 *
 * @detail Backend module should implement this Backend interface.
 *         Npu daemon will load this class symbol at runtime.
 */
class Backend
{
public:
  virtual ~Backend() = default;

  virtual NpudStatus getVersion(std::string &version) = 0;
  virtual NpudStatus createContext(NpudDevice *device, int device_fd, int priority,
                                   NpudContext *ctx) = 0;
  virtual NpudStatus destroyContext(NpudDevice *device, NpudContext ctx) = 0;
  virtual NpudStatus createBuffer(NpudDevice *device, GenericBuffer *buffer) = 0;
  virtual NpudStatus destroyBuffer(NpudDevice *device, GenericBuffer *buffer) = 0;
  // TODO Support to register model from buffer
  virtual NpudStatus registerModel(NpudDevice *device, const std::string &modelPath,
                                   ModelID *modelId) = 0;
  virtual NpudStatus unregisterModel(NpudDevice *device, ModelID modelId) = 0;
  virtual NpudStatus createRequest(NpudDevice *device, ModelID modelId, RequestID *requestId) = 0;
  virtual NpudStatus destroyRequest(NpudDevice *device, RequestID requestId) = 0;
  virtual NpudStatus setRequestData(NpudDevice *device, RequestID requestId,
                                    InputBuffers *input_bufs, TensorDataInfo *in_info,
                                    OutputBuffers *output_bufs, TensorDataInfo *out_info) = 0;
  virtual NpudStatus submitRequest(NpudDevice *device, RequestID requestId) = 0;
};

// std::string allocateSymbol("allocate");
// std::string deallocateSymbol("deallocate");

typedef Backend *(*NpudAlloc)();
typedef void (*NpudDealloc)(Backend *);

} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_BACKEND_H__
