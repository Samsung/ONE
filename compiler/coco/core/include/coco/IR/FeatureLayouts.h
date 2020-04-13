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

#ifndef __COCO_IR_FEATURE_LAYOUTS_H__
#define __COCO_IR_FEATURE_LAYOUTS_H__

#include "coco/IR/FeatureLayout.h"

#include <nncc/core/ADT/feature/Layout.h>

#include <vector>
#include <memory>

namespace coco
{
namespace FeatureLayouts
{

/**
 * @brief BCHW Feature Layout
 */
class BCHW final : public FeatureLayout
{
private:
  BCHW(const FeatureShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  static const FeatureLayout::ID *uid(void);
  const FeatureLayout::ID *id(void) const override { return uid(); }

  const FeatureShape &shape(void) const override { return _shape; }

  ElemID at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const override;

private:
  FeatureShape _shape;

public:
  static std::unique_ptr<BCHW> create(const nncc::core::ADT::feature::Shape &shape);
};

/**
 * @brief BHWC Feature Layout
 */
class BHWC : public coco::FeatureLayout
{
private:
  BHWC(const FeatureShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  static const FeatureLayout::ID *uid(void);
  const FeatureLayout::ID *id(void) const override { return uid(); }

  const FeatureShape &shape(void) const override { return _shape; }

  coco::ElemID at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const override;

private:
  FeatureShape _shape;

public:
  static std::unique_ptr<BHWC> create(const nncc::core::ADT::feature::Shape &shape);
  static std::unique_ptr<BHWC> create(const FeatureShape &shape);
};

/**
 * @brief BC (Channel-wise Channel-major) Feature Layout
 *
 * 1. A layout is said to be channel-wise if the following holds:
 *
 *  For each pair of valid feature index I and J,
 *    at(I) == at(J) if batch(I) == batch(J) and channel(I) == channel(J)
 *
 * 2. A layout is said to be channel-major if the followings hold:
 *
 *   For each pair of valid feature index I and J,
 *    at(I) + 1 == at(J) if batch(I) == batch(J) and channel(I) + 1 == channel(J)
 *
 *   For each pair of valid feature index I and J,
 *    at(I) + 1 == at(J) if batch(I) + 1 == batch(J), channel(I) == depth - 1, and channel(J) == 0
 */
class BC : public coco::FeatureLayout
{
private:
  BC(const FeatureShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  static const FeatureLayout::ID *uid(void);
  const FeatureLayout::ID *id(void) const override { return uid(); }

  const FeatureShape &shape(void) const override { return _shape; }

  coco::ElemID at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const override;

private:
  FeatureShape _shape;

public:
  static std::unique_ptr<BC> create(const nncc::core::ADT::feature::Shape &shape);
};

/**
 * @brief Generic Feature Layout
 */
class Generic final : public FeatureLayout
{
private:
  Generic(const FeatureShape &shape);

public:
  static const FeatureLayout::ID *uid(void);
  const FeatureLayout::ID *id(void) const override { return uid(); }

  const FeatureShape &shape(void) const override { return _shape; }

  ElemID &at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col);
  ElemID at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const override;

  void reorder(const nncc::core::ADT::feature::Layout &l);

private:
  uint32_t offset(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const;

private:
  FeatureShape _shape;

private:
  std::vector<ElemID> _content;

public:
  static std::unique_ptr<Generic> create(const nncc::core::ADT::feature::Shape &shape);
};

} // namespace FeatureLayouts
} // namespace coco

#endif // __COCO_IR_FEATURE_LAYOUTS_H__
