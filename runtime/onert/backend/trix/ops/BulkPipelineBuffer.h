/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEBUFFER_H__
#define __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEBUFFER_H__

#include <memory>
#include <cstdio>
#include <stdexcept>
#include <libnpuhost.h>

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{

class BulkPipelineBuffer
{
public:
  enum class BufferType
  {
    PROGRAM, // Program buffer (contiguous memory)
    WEIGHT   // Weight buffer (non-contiguous memory possible)
  };

  enum class MemoryType
  {
    DMABUF_CONT, // Contiguous DMA buffer
    DMABUF_IOMMU // IOMMU DMA buffer
  };

public:
  BulkPipelineBuffer(BufferType type, size_t size, int device_id);
  ~BulkPipelineBuffer();

  // Disallow copy/move due to unique resource
  BulkPipelineBuffer(const BulkPipelineBuffer &) = delete;
  BulkPipelineBuffer &operator=(const BulkPipelineBuffer &) = delete;
  BulkPipelineBuffer(BulkPipelineBuffer &&other) noexcept;
  BulkPipelineBuffer &operator=(BulkPipelineBuffer &&other) noexcept;

  // Buffer management functions
  void allocate();
  void deallocate();
  void *data() const;
  size_t size() const;
  int dmabufFd() const;

  generic_buffer *getGenericBuffer() { return _buffer; }
  const generic_buffer *getGenericBuffer() const { return _buffer; }

  // Data manipulation functions
  void fillFromFile(FILE *fp, size_t offset = 0);
  bool isReady() const;

private:
  size_t getAlignedSize(size_t size) const;
  MemoryType getMemoryType() const;

private:
  BufferType _type;
  size_t _size;
  int _device_id;
  int _dev_fd{-1};
  generic_buffer *_buffer{nullptr};
  MemoryType _memory_type;
};

class BulkPipelineBufferException : public std::runtime_error
{
public:
  explicit BulkPipelineBufferException(const std::string &msg) : std::runtime_error(msg) {}
};

class BufferAllocationException : public BulkPipelineBufferException
{
public:
  explicit BufferAllocationException(const std::string &msg) : BulkPipelineBufferException(msg) {}
};

class BufferOperationException : public BulkPipelineBufferException
{
public:
  explicit BufferOperationException(const std::string &msg) : BulkPipelineBufferException(msg) {}
};

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEBUFFER_H__
