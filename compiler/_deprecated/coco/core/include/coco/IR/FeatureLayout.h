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

#ifndef __COCO_IR_FEATURE_LAYOUT_H__
#define __COCO_IR_FEATURE_LAYOUT_H__

#include "coco/IR/ElemID.h"
#include "coco/IR/FeatureShape.h"

namespace coco
{

/**
 * @brief A FeatureLayout connects each feature index to a Bag element
 *
 * NOTE FeatureLayout is an immutable interface
 */
struct FeatureLayout
{
  struct ID
  {
    virtual ~ID() = default;
  };

  virtual ~FeatureLayout() = default;

  virtual const ID *id(void) const = 0;

  virtual const FeatureShape &shape(void) const = 0;

  uint32_t batch(void) const { return shape().batch(); }
  uint32_t depth(void) const { return shape().depth(); }
  uint32_t height(void) const { return shape().height(); }
  uint32_t width(void) const { return shape().width(); }

  virtual ElemID at(uint32_t b, uint32_t ch, uint32_t row, uint32_t col) const = 0;
};

} // namespace coco

#endif // __COCO_IR_FEATURE_LAYOUT_H__
