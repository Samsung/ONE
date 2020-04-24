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
#include <cassert>

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
  const uint8_t *base(void) const override
  {
    assert(_base != nullptr);
    return _base;
  }

  void release(void)
  {
    assert(_base != nullptr);
    delete[] _base;
    _base = nullptr;
  }

private:
  uint8_t *_base;
  size_t _size;
};

class ExternalData final : public Data
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

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_DATA_H__
