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

#ifndef __NNCC_CORE_ADT_FEATURE_BUFFER_H__
#define __NNCC_CORE_ADT_FEATURE_BUFFER_H__

#include "nncc/core/ADT/feature/View.h"

#include <vector>

namespace nncc
{
namespace core
{
namespace ADT
{
namespace feature
{

template <typename T> class Buffer final : public View<T>
{
public:
  explicit Buffer(const Shape &shape, const Layout &layout) : View<T>{shape, layout}
  {
    _buffer.resize(num_elements(shape));
  }

public:
  virtual T *base(void) { return _buffer.data(); }
  virtual const T *base(void) const { return _buffer.data(); }

private:
  std::vector<T> _buffer;
};

template <typename T, typename LayoutImpl> Buffer<T> make_buffer(const Shape &shape)
{
  return Buffer<T>{shape, LayoutImpl{}};
}

} // namespace feature
} // namespace ADT
} // namespace core
} // namespace nncc

#endif // __NNCC_CORE_ADT_FEATURE_BUFFER_H__
