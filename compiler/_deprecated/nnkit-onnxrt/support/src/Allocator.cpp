/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nnkit/support/onnx/Allocator.h"
#include "nnkit/support/onnx/Status.h"

#include <stdexcept>

namespace nnkit
{
namespace support
{
namespace onnx
{

Allocator::Allocator(void)
{
  OrtAllocator::version = ORT_API_VERSION;
  OrtAllocator::Alloc = [](OrtAllocator *this_, size_t size) {
    return static_cast<Allocator *>(this_)->Alloc(size);
  };
  OrtAllocator::Free = [](OrtAllocator *this_, void *p) {
    static_cast<Allocator *>(this_)->Free(p);
  };
  OrtAllocator::Info = [](const OrtAllocator *this_) {
    return static_cast<const Allocator *>(this_)->Info();
  };

  Status status;
  status = OrtCreateCpuAllocatorInfo(OrtDeviceAllocator, OrtMemTypeDefault, &_cpu_allocator_info);
  status.throwOnError();
}

Allocator::~Allocator(void) { OrtReleaseAllocatorInfo(_cpu_allocator_info); }

void *Allocator::Alloc(size_t size)
{
  // NOTE The extra_len is added to check resource leak.
  //
  //      This Alloc function will allocate the given size with extra_len.
  //      The first extra_len will save the allocated memory size and
  //      the user will use address from the allocated memory plus extra_len.
  //      The size value that saved in extra_len is used to Free function
  //      to check resource leak. The size value uses in _memory_inuse.
  constexpr size_t extra_len = sizeof(size_t);
  _memory_inuse.fetch_add(size += extra_len);
  void *p = ::malloc(size);
  *(size_t *)p = size;
  return (char *)p + extra_len;
}

void Allocator::Free(void *p)
{
  constexpr size_t extra_len = sizeof(size_t);
  if (!p)
    return;
  p = (char *)p - extra_len;
  size_t len = *(size_t *)p;
  _memory_inuse.fetch_sub(len);
  return ::free(p);
}

const OrtAllocatorInfo *Allocator::Info(void) const { return _cpu_allocator_info; }

void Allocator::LeakCheck(void)
{
  if (_memory_inuse.load())
  {
    throw std::runtime_error{"memory leak!!!"};
  }
}

} // namespace onnx
} // namespace support
} // namespace nnkit
