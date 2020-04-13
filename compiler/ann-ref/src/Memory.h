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

#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <cstdint>
#include <cstddef>

// Represents a memory region.
struct Memory
{
  Memory() = default;
  virtual ~Memory() = default;

  // Disallow copy semantics to ensure the runtime object can only be freed
  // once. Copy semantics could be enabled if some sort of reference counting
  // or deep-copy system for runtime objects is added later.
  Memory(const Memory &) = delete;
  Memory &operator=(const Memory &) = delete;

  // Returns a pointer to the underlying memory of this memory object.
  virtual int getPointer(uint8_t **buffer) const = 0;
  virtual bool validateSize(uint32_t offset, uint32_t length) const = 0;
};

class MappedMemory final : public Memory
{
public:
  MappedMemory() = default;

public:
  ~MappedMemory();

public:
  // Create the native_handle based on input size, prot, and fd.
  // Existing native_handle will be deleted, and mHidlMemory will wrap
  // the newly created native_handle.
  int set(size_t size, int prot, int fd, size_t offset);

public:
  int getPointer(uint8_t **buffer) const override;
  bool validateSize(uint32_t offset, uint32_t length) const override;

private:
  uint8_t *_base = nullptr;
  size_t _size = 0;
};

// Represents a memory region.
class AllocatedMemory : public Memory
{
public:
  AllocatedMemory() = default;
  virtual ~AllocatedMemory() = default;

public:
  virtual int create(uint32_t size) = 0;

public:
  // Returns a pointer to the underlying memory of this memory object.
  virtual int getPointer(uint8_t **buffer) const = 0;
  virtual bool validateSize(uint32_t offset, uint32_t length) const = 0;
};

class PrivateMemory final : public AllocatedMemory
{
public:
  PrivateMemory() = default;
  ~PrivateMemory();

public:
  // Disallow copy semantics to ensure the runtime object can only be freed
  // once. Copy semantics could be enabled if some sort of reference counting
  // or deep-copy system for runtime objects is added later.
  PrivateMemory(const PrivateMemory &) = delete;
  PrivateMemory &operator=(const PrivateMemory &) = delete;

public:
  virtual int create(uint32_t size);

public:
  // Returns a pointer to the underlying memory of this memory object.
  virtual int getPointer(uint8_t **buffer) const;
  virtual bool validateSize(uint32_t offset, uint32_t length) const;

private:
  uint8_t *_base = nullptr;
  size_t _size = 0;
};

#endif // __MEMORY_H__
