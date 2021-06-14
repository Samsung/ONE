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

#include "Buffer.h"

#include <string>

#include "ClContext.h"
#include "DataType.h"
#include "GpuObject.h"
#include "Util.h"
#include "Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace
{

absl::Status CreateBuffer(size_t size_in_bytes, bool gpu_read_only, const void *data,
                          CLContext *context, Buffer *result)
{
  cl_mem buffer;
  RETURN_IF_ERROR(CreateCLBuffer(context->context(), size_in_bytes, gpu_read_only,
                                 const_cast<void *>(data), &buffer));
  *result = Buffer(buffer, size_in_bytes);

  return absl::OkStatus();
}

} // namespace

BufferDescriptor::BufferDescriptor(BufferDescriptor &&desc)
  : GPUObjectDescriptor(std::move(desc)), element_type(desc.element_type),
    element_size(desc.element_size), memory_type(desc.memory_type),
    attributes(std::move(desc.attributes)), size(desc.size), data(std::move(desc.data))
{
}

BufferDescriptor &BufferDescriptor::operator=(BufferDescriptor &&desc)
{
  if (this != &desc)
  {
    std::swap(element_type, desc.element_type);
    std::swap(element_size, desc.element_size);
    std::swap(memory_type, desc.memory_type);
    attributes = std::move(desc.attributes);
    std::swap(size, desc.size);
    data = std::move(desc.data);
    GPUObjectDescriptor::operator=(std::move(desc));
  }
  return *this;
}

void BufferDescriptor::Release() { data.clear(); }

GPUResources BufferDescriptor::GetGPUResources() const
{
  GPUResources resources;
  GPUBufferDescriptor desc;
  desc.data_type = element_type;
  desc.access_type = access_type_;
  desc.element_size = element_size;
  desc.memory_type = memory_type;
  desc.attributes = attributes;
  resources.buffers.push_back({"buffer", desc});
  return resources;
}

absl::Status BufferDescriptor::CreateGPUObject(CLContext *context, GPUObjectPtr *result) const
{
  Buffer gpu_buffer;
  RETURN_IF_ERROR(gpu_buffer.CreateFromBufferDescriptor(*this, context));
  *result = absl::make_unique<Buffer>(std::move(gpu_buffer));
  return absl::OkStatus();
}

Buffer::Buffer(cl_mem buffer, size_t size_in_bytes) : buffer_(buffer), size_(size_in_bytes) {}

Buffer::Buffer(Buffer &&buffer) : buffer_(buffer.buffer_), size_(buffer.size_)
{
  buffer.buffer_ = nullptr;
  buffer.size_ = 0;
}

Buffer &Buffer::operator=(Buffer &&buffer)
{
  if (this != &buffer)
  {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(buffer_, buffer.buffer_);
  }
  return *this;
}

void Buffer::Release()
{
  if (buffer_)
  {
    clReleaseMemObject(buffer_);
    buffer_ = nullptr;
    size_ = 0;
  }
}

absl::Status Buffer::GetGPUResources(const GPUObjectDescriptor *obj_ptr,
                                     GPUResourcesWithValue *resources) const
{
  const auto *buffer_desc = dynamic_cast<const BufferDescriptor *>(obj_ptr);
  if (!buffer_desc)
  {
    return absl::InvalidArgumentError("Expected BufferDescriptor on input.");
  }

  resources->buffers.push_back({"buffer", buffer_});
  return absl::OkStatus();
}

absl::Status Buffer::CreateFromBufferDescriptor(const BufferDescriptor &desc, CLContext *context)
{
  bool read_only = desc.memory_type == MemoryType::CONSTANT;
  uint8_t *data_ptr = desc.data.empty() ? nullptr : const_cast<unsigned char *>(desc.data.data());
  size_ = desc.size;
  return CreateCLBuffer(context->context(), desc.size, read_only, data_ptr, &buffer_);
}

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, CLContext *context, Buffer *result)
{
  return CreateBuffer(size_in_bytes, true, nullptr, context, result);
}

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, const void *data, CLContext *context,
                                  Buffer *result)
{
  return CreateBuffer(size_in_bytes, true, data, context, result);
}

absl::Status CreateReadWriteBuffer(size_t size_in_bytes, CLContext *context, Buffer *result)
{
  return CreateBuffer(size_in_bytes, false, nullptr, context, result);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
