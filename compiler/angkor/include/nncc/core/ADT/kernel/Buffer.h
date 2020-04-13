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

#ifndef __NNCC_CORE_ADT_KERNEL_BUFFER_H__
#define __NNCC_CORE_ADT_KERNEL_BUFFER_H__

#include "nncc/core/ADT/kernel/View.h"
#include "nncc/core/ADT/kernel/ViewImpl.h"

#include <vector>

namespace nncc
{
namespace core
{
namespace ADT
{
namespace kernel
{

template <typename T> class Buffer final : public View<T>
{
public:
  explicit Buffer(const Shape &shape, const Layout &layout) : _impl{shape, layout}
  {
    _buffer.resize(num_elements(shape));
  }

public:
  T at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    return _impl.at(_buffer.begin(), nth, ch, row, col);
  }

public:
  T &at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) override
  {
    return _impl.at(_buffer.begin(), nth, ch, row, col);
  }

public:
  const Shape &shape(void) const override { return _impl.shape(); }

private:
  std::vector<T> _buffer;
  ViewImpl<T> _impl;
};

template <typename T, typename LayoutImpl> Buffer<T> make_buffer(const Shape &shape)
{
  return Buffer<T>{shape, LayoutImpl{}};
}

} // namespace kernel
} // namespace ADT
} // namespace core
} // namespace nncc

#endif // __NNCC_CORE_ADT_KERNEL_BUFFER_H__
