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

#ifndef MEMORY_POOL_FOR_SYMBOL_SEARCHER_INTERNALS_H
#define MEMORY_POOL_FOR_SYMBOL_SEARCHER_INTERNALS_H

#include <cstddef>
#include <cstdint>

// TODO this class possibly should be thread safe (or all symbols should be resolved at the start of
// application as alternative)
class MemoryPoolForSymbolSearcherInternals
{
  static constexpr size_t MAX_SIZE = 65536;

public:
  bool containsMemorySpaceStartedFromPointer(void *ptr) noexcept
  {
    return ptr >= _buffer && ptr < _buffer + MAX_SIZE;
  }

  // TODO this function should return alighned ptr to avoid potential problems
  void *allocate(size_t size) noexcept
  {
    if (isSpaceOfRequiredSizeNotAvailable(size))
    {
      // TODO need to signalize about error
    }

    uint8_t *ptr_to_memory_space_begin = _ptr_to_free_space_start;
    _ptr_to_free_space_start += size;
    _size_of_last_allocated_space = size;
    return ptr_to_memory_space_begin;
  }

  void deallocate(void *p) noexcept
  {
    if (p == _ptr_to_free_space_start - _size_of_last_allocated_space)
    {
      _ptr_to_free_space_start -= _size_of_last_allocated_space;
      _size_of_last_allocated_space = 0;
    }
  }

private:
  bool isSpaceOfRequiredSizeNotAvailable(size_t size)
  {
    uint8_t *ptr_to_the_free_space_after_allocation = _ptr_to_free_space_start + size;
    size_t size_of_reserved_space_after_allocation =
      ptr_to_the_free_space_after_allocation - _buffer;
    if (size_of_reserved_space_after_allocation >= MAX_SIZE)
    {
      return false;
    }

    return true;
  }

private:
  static uint8_t _buffer[MAX_SIZE];
  static uint8_t *volatile _ptr_to_free_space_start;
  static volatile size_t _size_of_last_allocated_space;
};

#endif // ! MEMORY_POOL_FOR_SYMBOL_SEARCHER_INTERNALS_H
