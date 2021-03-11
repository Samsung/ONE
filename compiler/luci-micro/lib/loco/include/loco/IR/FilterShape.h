/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_IR_FILTER_SHAPE_H__
#define __LOCO_IR_FILTER_SHAPE_H__

#include "loco/IR/Dimension.h"

namespace loco
{

/**
 * \brief Filter Shape
 *
 * This class describes the shape of filter, which is an input of 2D
 * convolutional operations (e.g. Convolution).
 *
 * count() refers to the number of 3D weight in a filter
 * depth() refers to the depth of 3D weights
 * height() refers to the height of 3D weights
 * width() refers to the width of 3D weights
 *
 * NOTE
 *
 * The definition of FilterShape is almost same as that of FeatureShape, but loco
 * distinguishes FeatureShape and FilterShape in class-level in order to prevent
 * potential errors by type check.
 */
class FilterShape final
{
public:
  FilterShape() = default;

public:
  const Dimension &count(void) const { return _count; }
  Dimension &count(void) { return _count; }

  const Dimension &depth(void) const { return _depth; }
  Dimension &depth(void) { return _depth; }

  const Dimension &height(void) const { return _height; }
  Dimension &height(void) { return _height; }

  const Dimension &width(void) const { return _width; }
  Dimension &width(void) { return _width; }

private:
  Dimension _count;
  Dimension _depth;
  Dimension _height;
  Dimension _width;
};

} // namespace loco

#endif // __LOCO_IR_FILTER_SHAPE_H__
