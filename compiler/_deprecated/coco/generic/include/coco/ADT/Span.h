/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __COCO_ADT_SPAN_H__
#define __COCO_ADT_SPAN_H__

#include <cstdint>
#include <cassert>

namespace coco
{

/**
 * @brief A Span is a non-owing reference to a memory chunk
 *
 * @note A Span DOES NOT OWN a memory chunk.
 */
template <typename T> class Span
{
public:
  Span(T *data, uint32_t size) : _data{data}, _size{size}
  {
    // DO NOTHING
  }

public:
  T *data(void) { return _data; }
  const T *data(void) const { return _data; }

public:
  uint32_t size(void) const { return _size; }

public:
  T &operator[](uint32_t n)
  {
    assert(n < _size);
    return *(_data + n);
  }

public:
  const T &operator[](uint32_t n) const
  {
    assert(n < _size);
    return *(_data + n);
  }

private:
  T *_data;
  uint32_t _size;
};

} // namespace coco

#endif // __COCO_ADT_SPAN_H__
