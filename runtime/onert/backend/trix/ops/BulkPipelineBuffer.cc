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

#include "BulkPipelineBuffer.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{

// FIXME: Using higher level API instead of raw API
struct trix_ioctl_hwmem
{
  int32_t type;
  uint64_t size;
  int32_t dbuf_fd;
} __attribute__((packed));

#define TRIX_IOCTL_HWMEM_ALLOC _IOW(136, 21, struct trix_ioctl_hwmem)
#define TRIX_IOCTL_HWMEM_DEALLOC _IOW(136, 22, struct trix_ioctl_hwmem)

BulkPipelineBuffer::BulkPipelineBuffer(BufferType type, size_t size, int device_id)
  : _type(type), _size(size), _device_id(device_id)
{
  // DO NOTHING
}

BulkPipelineBuffer::~BulkPipelineBuffer() { deallocate(); }

size_t BulkPipelineBuffer::size() const { return _buffer ? _buffer->size : 0; }

bool BulkPipelineBuffer::isReady() const { return _buffer && _buffer->addr != nullptr; }

void BulkPipelineBuffer::allocate()
{
  if (_buffer && _buffer->addr != nullptr)
  {
    // Already allocated
    return;
  }

  if (!_buffer)
  {
    _buffer = new generic_buffer{};
  }

  // Open the devbice
  char devname[16];
  snprintf(devname, sizeof(devname), "/dev/triv2-%d", _device_id);
  _dev_fd = open(devname, O_RDWR);
  if (_dev_fd < 0)
  {
    throw std::runtime_error("Failed to open NPU device: " + std::string(devname));
  }

  // Allocate a buffer
  struct trix_ioctl_hwmem hwmem;
  hwmem.type = (_type == BufferType::DMABUF_CONT) ? 0 : 1;
  hwmem.size = getAlignedSize(_size);

  _buffer->dmabuf = ioctl(_dev_fd, TRIX_IOCTL_HWMEM_ALLOC, &hwmem);
  if (_buffer->dmabuf < 0)
  {
    close(_dev_fd);
    _dev_fd = -1;
    throw std::runtime_error("Failed to allocate DMA buffer, size: " + std::to_string(hwmem.size));
  }

  // Mapping the buffer
  _buffer->addr = mmap(nullptr, hwmem.size, PROT_READ | PROT_WRITE, MAP_SHARED, _buffer->dmabuf, 0);
  if (_buffer->addr == MAP_FAILED)
  {
    close(_buffer->dmabuf);
    close(_dev_fd);
    _buffer->dmabuf = -1;
    _dev_fd = -1;
    _buffer->addr = nullptr;
    throw std::runtime_error("Failed to mmap DMA buffer");
  }

  _buffer->size = _size;
  _buffer->type = BUFFER_DMABUF;
}

void BulkPipelineBuffer::deallocate()
{
  if (!_buffer)
  {
    return;
  }

  if (_buffer->addr != nullptr)
  {
    size_t aligned_sz = getAlignedSize(_buffer->size);
    munmap(_buffer->addr, aligned_sz);
    _buffer->addr = nullptr;
  }

  if (_buffer->dmabuf >= 0)
  {
    struct trix_ioctl_hwmem hwmem;
    hwmem.dbuf_fd = _buffer->dmabuf;
    ioctl(_dev_fd, TRIX_IOCTL_HWMEM_DEALLOC, &hwmem);
    close(_buffer->dmabuf);
    _buffer->dmabuf = -1;
  }

  if (_dev_fd >= 0)
  {
    close(_dev_fd);
    _dev_fd = -1;
  }

  delete _buffer;
  _buffer = nullptr;
}

void BulkPipelineBuffer::fillFromFile(FILE *fp, size_t offset)
{
  if (!isReady())
  {
    throw std::runtime_error("Buffer is not allocated");
  }

  if (!fp)
  {
    throw std::runtime_error("Invalid file pointer");
  }

  if (fseek(fp, static_cast<long>(offset), SEEK_SET) != 0)
  {
    throw std::runtime_error("Failed to seek file to offset: " + std::to_string(offset));
  }

  if (fread(_buffer->addr, _buffer->size, 1, fp) != 1)
  {
    throw std::runtime_error("Failed to read " + std::to_string(_buffer->size) +
                             " bytes from file");
  }
}

size_t BulkPipelineBuffer::getAlignedSize(size_t size) const
{
  // 4 KB (= Page size) aligned size
  constexpr size_t _4KB_M_1 = (1 << 12) - 1;
  return (size + _4KB_M_1) & ~_4KB_M_1;
}

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert
