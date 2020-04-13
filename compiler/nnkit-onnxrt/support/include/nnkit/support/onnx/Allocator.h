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

#ifndef __NNKIT_SUPPORT_ONNX_ALLOCATOR_H__
#define __NNKIT_SUPPORT_ONNX_ALLOCATOR_H__

#include <onnxruntime_c_api.h>

#include <atomic>

namespace nnkit
{
namespace support
{
namespace onnx
{

class Allocator final : public OrtAllocator
{
public:
  Allocator(void);
  ~Allocator(void);

  void *Alloc(size_t size);
  void Free(void *p);
  const OrtAllocatorInfo *Info(void) const;

  void LeakCheck(void);

  // Disallow copying
  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;

private:
  std::atomic<size_t> _memory_inuse{0};
  OrtAllocatorInfo *_cpu_allocator_info;
};

} // namespace onnx
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_ONNX_ALLOCATOR_H__
