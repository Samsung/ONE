/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <NeuralNetworks.h>
#include <sys/mman.h>

#include "ANeuralNetworksMemory.h"

//
// ANeuralNetworksMemory
//
ANeuralNetworksMemory::ANeuralNetworksMemory(size_t size, int protect, int fd, size_t offset)
{
  _base = reinterpret_cast<uint8_t *>(mmap(nullptr, size, protect, MAP_PRIVATE, fd, offset));
  _size = size;
}

ANeuralNetworksMemory::~ANeuralNetworksMemory() { munmap(reinterpret_cast<void *>(_base), _size); }

bool ANeuralNetworksMemory::vaildAccess(size_t offset, size_t length) const
{
  if ((offset >= _size) || (length > _size))
  {
    return false;
  }

  if ((offset + length) >= _size)
  {
    return false;
  }

  return true;
}
