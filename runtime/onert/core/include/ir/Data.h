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

#ifndef __ONERT_IR_DATA_H__
#define __ONERT_IR_DATA_H__

#include <algorithm>
#include <sys/mman.h>

namespace onert
{
namespace ir
{

struct Data
{
  virtual ~Data() = default;

  virtual size_t size(void) const = 0;
  virtual const uint8_t *base(void) const = 0;
};

class CachedData final : public Data
{
public:
  CachedData(const uint8_t *base, size_t size) : _base{new uint8_t[size]}, _size{size}
  {
    std::copy(base, base + size, _base);
  }

public:
  ~CachedData() { delete[] _base; }

public:
  size_t size(void) const override { return _size; }
  const uint8_t *base(void) const override { return _base; }

private:
  uint8_t *_base;
  size_t _size;
};

class ExternalData : public Data
{
public:
  ExternalData(const uint8_t *base, size_t size) : _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  size_t size(void) const override { return _size; }
  const uint8_t *base(void) const override { return _base; }

private:
  const uint8_t *_base;
  const size_t _size;
};

class MMapedData final : public ExternalData
{
public:
  MMapedData(uint8_t *mmap_base, int32_t page_size, const uint8_t *base, size_t size)
      : ExternalData(base, size), _mmap_base{mmap_base}, _page_size{page_size}
  {
    using std::ptrdiff_t;
    // Calculate offset from base address of mapped region
    ptrdiff_t unaligned_offset_start = base - _mmap_base;
    ptrdiff_t unaligned_offset_end = unaligned_offset_start + size;

    // Calculated aligned offset from base address of mapped region
    // munmap accepts memory address which is a multiple of the pagesize
    _offset_start = ((unaligned_offset_start + (_page_size - 1)) / _page_size) * _page_size;
    ptrdiff_t aligned_offset_end = (unaligned_offset_end / _page_size) * _page_size;

    _area_size = aligned_offset_end - _offset_start;
  }

public:
  ~MMapedData()
  {
    if (_area_size > 0)
    {
      munmap(const_cast<uint8_t *>(_mmap_base) + _offset_start, _area_size);
    }
  }

private:
  const uint8_t *_mmap_base;
  const int32_t _page_size;
  std::ptrdiff_t _offset_start;
  size_t _area_size;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_DATA_H__
