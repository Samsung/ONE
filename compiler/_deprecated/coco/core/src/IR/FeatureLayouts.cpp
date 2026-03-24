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

#include "coco/IR/FeatureLayouts.h"

#include <nncc/core/ADT/feature/CHWLayout.h>
#include <nncc/core/ADT/feature/HWCLayout.h>

#include <cassert>

using namespace nncc::core::ADT::feature;

//
// BCHW Layout
//
namespace coco
{
namespace FeatureLayouts
{

const FeatureLayout::ID *BCHW::uid(void)
{
  struct LayoutID final : public FeatureLayout::ID
  {
  };
  static LayoutID id;
  return &id;
}

ElemID BCHW::at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const
{
  static CHWLayout l;

  uint32_t offset = 0;
  offset += b * num_elements(_shape);
  offset += l.offset(_shape, ch, row, col);
  return ElemID{offset};
}

std::unique_ptr<BCHW> BCHW::create(const nncc::core::ADT::feature::Shape &shape)
{
  // NOTE It is impossible to use make_unique here as the constructor is private
  return std::unique_ptr<BCHW>{new BCHW{FeatureShape{shape}}};
}

} // namespace FeatureLayouts
} // namespace coco

//
// BHWC Layout
//
namespace coco
{
namespace FeatureLayouts
{

const FeatureLayout::ID *BHWC::uid(void)
{
  struct LayoutID final : public FeatureLayout::ID
  {
  };
  static LayoutID id;
  return &id;
}

ElemID BHWC::at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const
{
  static HWCLayout l;

  uint32_t offset = 0;
  offset += b * num_elements(_shape);
  offset += l.offset(_shape, ch, row, col);

  return ElemID{offset};
}

std::unique_ptr<BHWC> BHWC::create(const nncc::core::ADT::feature::Shape &shape)
{
  // NOTE It is impossible to use make_unique here as the constructor is private
  return std::unique_ptr<BHWC>{new BHWC{FeatureShape{shape}}};
}

std::unique_ptr<BHWC> BHWC::create(const FeatureShape &shape)
{
  // NOTE It is impossible to use make_unique here as the constructor is private
  return std::unique_ptr<BHWC>{new BHWC{shape}};
}

} // namespace FeatureLayouts
} // namespace coco

//
// BC: Channel-major Channel-wise Layout
//
namespace coco
{
namespace FeatureLayouts
{

const FeatureLayout::ID *BC::uid(void)
{
  struct LayoutID final : public FeatureLayout::ID
  {
  };
  static LayoutID id;
  return &id;
}

// NOTE BC layout ignores row/col as its name suggests
ElemID BC::at(uint32_t b, uint32_t ch, uint32_t /*row*/, uint32_t /*col*/) const
{
  assert(b < shape().batch());

  uint32_t offset = 0;

  offset += b * _shape.depth();
  offset += ch;

  return ElemID{offset};
}

std::unique_ptr<BC> BC::create(const nncc::core::ADT::feature::Shape &shape)
{
  // NOTE It is impossible to use make_unique here as the constructor is private
  return std::unique_ptr<BC>{new BC{FeatureShape{shape}}};
}

} // namespace FeatureLayouts
} // namespace coco

//
// Generic Layout
//
namespace coco
{
namespace FeatureLayouts
{

Generic::Generic(const FeatureShape &shape) : _shape{shape}
{
  _content.resize(_shape.batch() * num_elements(_shape));
}

const FeatureLayout::ID *Generic::uid(void)
{
  struct LayoutID final : public FeatureLayout::ID
  {
  };
  static LayoutID id;
  return &id;
}

uint32_t Generic::offset(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const
{
  static nncc::core::ADT::feature::CHWLayout l{};

  uint32_t res = 0;

  res += b * num_elements(_shape);
  res += l.offset(shape(), ch, row, col);

  return res;
}

ElemID &Generic::at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col)
{
  return _content.at(offset(b, ch, row, col));
}

ElemID Generic::at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const
{
  return _content.at(offset(b, ch, row, col));
}

void Generic::reorder(const nncc::core::ADT::feature::Layout &l)
{
  assert(shape().batch() == 1);

  for (uint32_t ch = 0; ch < shape().depth(); ++ch)
  {
    for (uint32_t row = 0; row < shape().height(); ++row)
    {
      for (uint32_t col = 0; col < shape().width(); ++col)
      {
        at(0, ch, row, col) = ElemID{l.offset(shape(), ch, row, col)};
      }
    }
  }
}

std::unique_ptr<Generic> Generic::create(const nncc::core::ADT::feature::Shape &shape)
{
  // NOTE It is impossible to use make_unique here as the constructor is private
  return std::unique_ptr<Generic>{new Generic{shape}};
}

} // namespace FeatureLayouts
} // namespace coco
