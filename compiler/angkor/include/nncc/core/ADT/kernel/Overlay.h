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

#ifndef __NNCC_CORE_ADT_KERNEL_OVERLAY_H__
#define __NNCC_CORE_ADT_KERNEL_OVERLAY_H__

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

template <typename T, typename InputIt> class Overlay final : public View<T>
{
public:
  explicit Overlay(const Shape &shape, const Layout &layout, InputIt it)
    : _impl{shape, layout}, _it{it}
  {
    // DO NOTHING
  }

public:
  T at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    return _impl.at(_it, nth, ch, row, col);
  }

public:
  T &at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) override
  {
    return _impl.at(_it, nth, ch, row, col);
  }

public:
  const Shape &shape(void) const override { return _impl.shape(); }

private:
  InputIt const _it;
  ViewImpl<T> _impl;
};

template <typename T, typename LayoutImpl> struct OverlayFactory
{
  template <typename InputIt> static Overlay<T, InputIt> make(const Shape &shape, InputIt it)
  {
    return Overlay<T, InputIt>{shape, LayoutImpl{}, it};
  }
};

template <typename T, typename LayoutImpl> Overlay<T, T *> make_overlay(const Shape &shape, T *base)
{
  return OverlayFactory<T, LayoutImpl>::make(shape, base);
}

} // namespace kernel
} // namespace ADT
} // namespace core
} // namespace nncc

#endif // __NNCC_CORE_ADT_KERNEL_OVERLAY_H__
