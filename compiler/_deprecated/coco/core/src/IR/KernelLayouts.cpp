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

#include "coco/IR/KernelLayouts.h"

#include <nncc/core/ADT/kernel/NCHWLayout.h>
#include <nncc/core/ADT/kernel/NHWCLayout.h>

#include <cassert>

using namespace nncc::core::ADT::kernel;

using nncc::core::ADT::kernel::num_elements;
using nncc::core::ADT::kernel::Shape;

//
// NCHW Layout
//
namespace coco
{
namespace KernelLayouts
{

const KernelLayout::ID *NCHW::uid(void)
{
  struct LayoutID final : public KernelLayout::ID
  {
  };
  static LayoutID id;
  return &id;
}

ElemID NCHW::at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const
{
  static NCHWLayout l;
  return ElemID{l.offset(_shape, n, ch, row, col)};
}

std::unique_ptr<NCHW> NCHW::create(const nncc::core::ADT::kernel::Shape &shape)
{
  // NOTE It is impossible to use make_unique here as the constructor is private
  return std::unique_ptr<NCHW>{new NCHW{shape}};
}

} // namespace KernelLayouts
} // namespace coco

//
// NHWC Layout
//
namespace coco
{
namespace KernelLayouts
{

const KernelLayout::ID *NHWC::uid(void)
{
  struct LayoutID final : public KernelLayout::ID
  {
  };
  static LayoutID id;
  return &id;
}

ElemID NHWC::at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const
{
  static NHWCLayout l;
  return ElemID{l.offset(_shape, n, ch, row, col)};
}

std::unique_ptr<NHWC> NHWC::create(const nncc::core::ADT::kernel::Shape &shape)
{
  // NOTE It is impossible to use make_unique here as the constructor is private
  return std::unique_ptr<NHWC>{new NHWC{shape}};
}

} // namespace KernelLayouts
} // namespace coco

//
// Generic Layout
//
namespace
{

nncc::core::ADT::kernel::NCHWLayout l;

} // namespace

namespace coco
{
namespace KernelLayouts
{

Generic::Generic(const nncc::core::ADT::kernel::Shape &shape) : _shape{shape}
{
  _content.resize(num_elements(_shape));
}

const KernelLayout::ID *Generic::uid(void)
{
  struct LayoutID final : public KernelLayout::ID
  {
  };
  static LayoutID id;
  return &id;
}

ElemID &Generic::at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col)
{
  return _content.at(l.offset(_shape, n, ch, row, col));
}

ElemID Generic::at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const
{
  return _content.at(l.offset(_shape, n, ch, row, col));
}

void Generic::reorder(const nncc::core::ADT::kernel::Layout &l)
{
  for (uint32_t n = 0; n < shape().count(); ++n)
  {
    for (uint32_t ch = 0; ch < shape().depth(); ++ch)
    {
      for (uint32_t row = 0; row < shape().height(); ++row)
      {
        for (uint32_t col = 0; col < shape().width(); ++col)
        {
          at(n, ch, row, col) = ElemID{l.offset(shape(), n, ch, row, col)};
        }
      }
    }
  }
}

std::unique_ptr<Generic> Generic::create(const Shape &shape)
{
  return std::unique_ptr<Generic>{new Generic{shape}};
}

} // namespace KernelLayouts
} // namespace coco
