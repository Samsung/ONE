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

#ifndef __NNCC_CORE_ADT_FEATURE_VIEW_H__
#define __NNCC_CORE_ADT_FEATURE_VIEW_H__

#include "nncc/core/ADT/feature/Shape.h"
#include "nncc/core/ADT/feature/Reader.h"
#include "nncc/core/ADT/feature/Accessor.h"
#include "nncc/core/ADT/feature/Layout.h"

namespace nncc
{
namespace core
{
namespace ADT
{
namespace feature
{

template <typename T> class View : public Reader<T>, public Accessor<T>
{
public:
  explicit View(const Shape &shape, const Layout &layout) : _shape{shape}, _layout{layout}
  {
    // DO NOTHING
  }

public:
  virtual T *base(void) = 0;
  virtual const T *base(void) const = 0;

public:
  T at(uint32_t ch, uint32_t row, uint32_t col) const override final
  {
    return *(base() + _layout.offset(_shape, ch, row, col));
  }

public:
  T &at(uint32_t ch, uint32_t row, uint32_t col) override final
  {
    return *(base() + _layout.offset(_shape, ch, row, col));
  }

public:
  const Shape &shape(void) const { return _shape; }

private:
  const Shape _shape;
  const Layout _layout;
};

} // namespace feature
} // namespace ADT
} // namespace core
} // namespace nncc

#endif // __NNCC_CORE_ADT_FEATURE_VIEW_H__
