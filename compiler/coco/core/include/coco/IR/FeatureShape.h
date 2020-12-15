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

#ifndef __COCO_IR_FEATURE_SHAPE_H__
#define __COCO_IR_FEATURE_SHAPE_H__

#include <nncc/core/ADT/feature/Shape.h>

namespace coco
{

/**
 * @brief The shape of a feature map
 *
 * TODO Implement coco's own FeatureShape without "nncc::core::ADT::feature::Shape"
 */
class FeatureShape : public nncc::core::ADT::feature::Shape
{
public:
  FeatureShape(uint32_t depth, uint32_t height, uint32_t width)
    : Shape{depth, height, width}, _batch{1}
  {
    // DO NOTHING
  }

  FeatureShape(uint32_t batch, uint32_t depth, uint32_t height, uint32_t width)
    : Shape{depth, height, width}, _batch{batch}
  {
    // DO NOTHING
  }

  FeatureShape(const nncc::core::ADT::feature::Shape &shape) : Shape{shape}, _batch{1}
  {
    // DO NOTHING
  }

public:
  uint32_t batch(void) const { return _batch; }

private:
  uint32_t _batch;
};

static inline bool operator==(const FeatureShape &lhs, const FeatureShape &rhs)
{
  return (lhs.batch() == rhs.batch()) && (lhs.depth() == rhs.depth()) &&
         (lhs.height() == rhs.height()) && (lhs.width() == rhs.width());
}

static inline bool operator!=(const FeatureShape &lhs, const FeatureShape &rhs)
{
  return !(lhs == rhs);
}

} // namespace coco

#endif // __COCO_IR_FEATURE_SHAPE_H__
