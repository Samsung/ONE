/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#define LOG_TAG "Memory"

#include "Memory.h"
#include "NeuralNetworks.h" // ANEURALNETWORKS_XXX

#include <sys/mman.h>

MappedMemory::~MappedMemory()
{
  if (_base)
  {
    munmap(_base, _size);
  }
}

int MappedMemory::set(size_t size, int prot, int fd, size_t offset)
{
#if 0
  if (fd < 0)
  {
    LOG(ERROR) << "ANeuralNetworksMemory_createFromFd invalid fd " << fd;
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  if (size == 0 || fd < 0)
  {
    LOG(ERROR) << "Invalid size or fd";
    return ANEURALNETWORKS_BAD_DATA;
  }
  int dupfd = dup(fd);
  if (dupfd == -1)
  {
    LOG(ERROR) << "Failed to dup the fd";
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
#endif
  void * const base = mmap(nullptr, size, prot, MAP_PRIVATE, fd, offset);

  if (base == MAP_FAILED)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  _base = static_cast<uint8_t *>(base);
  _size = size;

  return ANEURALNETWORKS_NO_ERROR;
}

int MappedMemory::getPointer(uint8_t **buffer) const
{
  *buffer = _base;
  return ANEURALNETWORKS_NO_ERROR;
}

bool MappedMemory::validateSize(uint32_t offset, uint32_t length) const
{
  return true;
}

PrivateMemory::~PrivateMemory()
{
  if (_base)
  {
    delete[] _base;
  }
}

int PrivateMemory::create(uint32_t size)
{
  auto base = new uint8_t[size];

  // TODO Check allocation failure
  _base = base;
  _size = size;

  return ANEURALNETWORKS_NO_ERROR;
}

int PrivateMemory::getPointer(uint8_t **buffer) const
{
  *buffer = _base;
  return ANEURALNETWORKS_NO_ERROR;
}

bool PrivateMemory::validateSize(uint32_t offset, uint32_t length) const
{
  return true;
}
