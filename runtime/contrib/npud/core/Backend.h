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

#include <cstdint>
#include <string>
#include <vector>

namespace npud
{
namespace core
{

#define NPU_TENSOR_MAX (16)

/**
 * @brief Npu model ID.
 *
 */
using ModelID = uint32_t;

/**
 * @brief Npu request ID.
 *
 */
using RequestID = uint32_t;

/**
 * @brief Npu buffer type
 *
 */
enum BufferTypes
{
  NPU_BUFFER_MAPPED,   /**< buffer is a memory-mapped ptr */
  NPU_BUFFER_DMABUF,   /**< buffer is a dmabuf fd, representing contiguous memory */
  NPU_BUFFER_UNDEFINED /**< buffer type is undefined */
};

/**
 * @brief Various kinds of buffer supported for input/output/model.
 *
 */
struct GenericBuffer
{
  struct
  {             /** NPU_BUFFER_MAPPED/DMABUF */
    void *addr; /**< Mapped address of the buffer */
    struct
    {                  /** NPU_BUFFER_DMABUF only */
      int dmabuf;      /**< The dma-buf fd handle of the memory allocated */
      uint64_t offset; /**< Offset to be applied to the base memory address */
    };
  };
  uint64_t size;    /**< The size of the buffer in bytes */
  BufferTypes type; /**< Type of memory in this buffer */
};

/**
 * @brief Npu generic buffer array.
 *
 */
struct GenericBuffers
{
  uint32_t numBuffers;
  GenericBuffer buffers[NPU_TENSOR_MAX];
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
 * @brief Npu tensor data info array.
 *
 */
struct TensorDataInfos
{
  uint32_t numInfos;
  TensorDataInfo infos[NPU_TENSOR_MAX];
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
  NPU_STATUS_ERROR_INVALID_MODEL,
  NPU_STATUS_ERROR_INVALID_DATA,
};

/**
 * @brief Npu context definition
 *
 * @param models The model lists.
 * @param requests The request lists.
 * @param defaultCore The core number to be used by default.
 */
struct NpuContext
{
  std::vector<ModelID> models;
  std::vector<RequestID> requests;
  int defaultCore;
};

/**
 * @brief Npu backend interface
 *
 * Backend module should implement this Backend interface.
 * Npu daemon will load this class symbol at runtime.
 */
class Backend
{
public:
  virtual ~Backend() = default;

  virtual NpuStatus getVersion(std::string &version) = 0;
  virtual NpuStatus createContext(int deviceId, int priority, NpuContext **ctx) = 0;
  virtual NpuStatus destroyContext(NpuContext *ctx) = 0;
  virtual NpuStatus createBuffer(NpuContext *ctx, GenericBuffer *buffer) = 0;
  virtual NpuStatus destroyBuffer(NpuContext *ctx, GenericBuffer *buffer) = 0;
  // TODO Support to register model from buffer
  virtual NpuStatus registerModel(NpuContext *ctx, const std::string &modelPath,
                                  ModelID *modelId) = 0;
  virtual NpuStatus unregisterModel(NpuContext *ctx, ModelID modelId) = 0;
  virtual NpuStatus createRequest(NpuContext *ctx, ModelID modelId, RequestID *requestId) = 0;
  virtual NpuStatus destroyRequest(NpuContext *ctx, RequestID requestId) = 0;
  virtual NpuStatus setRequestData(NpuContext *ctx, RequestID requestId, InputBuffers *inputBufs,
                                   TensorDataInfos *inputInfos, OutputBuffers *outputBufs,
                                   TensorDataInfos *outputInfos) = 0;
  virtual NpuStatus submitRequest(NpuContext *ctx, RequestID requestId) = 0;
};

typedef Backend *(*NpuAlloc)();
typedef void (*NpuDealloc)(Backend *);

} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_BACKEND_H__
