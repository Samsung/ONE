/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_BUFFER_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_BUFFER_H__

#include "ClCommandQueue.h"
#include "ClContext.h"
#include "GpuObject.h"
#include "OpenclWrapper.h"
#include "DataType.h"
#include "Util.h"
#include "Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

struct BufferDescriptor : public GPUObjectDescriptor
{
  DataType element_type;
  int element_size;
  MemoryType memory_type = MemoryType::GLOBAL;
  std::vector<std::string> attributes;

  // optional
  int size = 0;
  std::vector<uint8_t> data;

  BufferDescriptor() = default;
  BufferDescriptor(const BufferDescriptor &) = default;
  BufferDescriptor &operator=(const BufferDescriptor &) = default;
  BufferDescriptor(BufferDescriptor &&desc);
  BufferDescriptor &operator=(BufferDescriptor &&desc);

  GPUResources GetGPUResources() const override;

  absl::Status CreateGPUObject(CLContext *context, GPUObjectPtr *result) const override;
  void Release() override;
};

// Buffer represent linear GPU data storage with arbitrary data format.
// Buffer is moveable but not copyable.
class Buffer : public GPUObject
{
public:
  Buffer() {} // just for using Buffer as a class members
  Buffer(cl_mem buffer, size_t size_in_bytes);

  // Move only
  Buffer(Buffer &&buffer);
  Buffer &operator=(Buffer &&buffer);
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  virtual ~Buffer() { Release(); }

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const { return size_; }

  cl_mem GetMemoryPtr() const { return buffer_; }

  // Writes data to a buffer. Data should point to a region that
  // has exact size in bytes as size_in_bytes(constructor parameter).
  template <typename T> absl::Status WriteData(CLCommandQueue *queue, const std::vector<T> *data);

  // Reads data from Buffer into CPU memory.
  template <typename T> absl::Status ReadData(CLCommandQueue *queue, std::vector<T> *result) const;

  absl::Status GetGPUResources(const GPUObjectDescriptor *obj_ptr,
                               GPUResourcesWithValue *resources) const override;

  absl::Status CreateFromBufferDescriptor(const BufferDescriptor &desc, CLContext *context);

private:
  void Release();

  cl_mem buffer_ = nullptr;
  size_t size_ = 0;
};

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, CLContext *context, Buffer *result);

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, const void *data, CLContext *context,
                                  Buffer *result);

absl::Status CreateReadWriteBuffer(size_t size_in_bytes, CLContext *context, Buffer *result);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_BUFFER_H__
