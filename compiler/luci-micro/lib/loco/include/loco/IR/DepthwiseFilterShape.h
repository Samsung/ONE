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

#ifndef __LOCO_IR_DEPTHWISE_FILTER_SHAPE_H__
#define __LOCO_IR_DEPTHWISE_FILTER_SHAPE_H__

#include "loco/IR/Dimension.h"

namespace loco
{

/**
 * @brief DepthwiseFilter Shape
 *
 * This class describes the shape of depthwise filter, which is an input of depthwise 2D
 * convolutional operation.
 *
 * depth() refers to expected channel depth of matching input
 * multiplier() refers to number of traverse for one input
 * height() refers to the height of 2D weights
 * width() refers to the width of 2D weights
 */
class DepthwiseFilterShape final
{
public:
  DepthwiseFilterShape() = default;

public:
  const Dimension &depth(void) const { return _depth; }
  Dimension &depth(void) { return _depth; }

  const Dimension &multiplier(void) const { return _multiplier; }
  Dimension &multiplier(void) { return _multiplier; }

  const Dimension &height(void) const { return _height; }
  Dimension &height(void) { return _height; }

  const Dimension &width(void) const { return _width; }
  Dimension &width(void) { return _width; }

private:
  Dimension _depth;
  Dimension _multiplier;
  Dimension _height;
  Dimension _width;
};

} // namespace loco

#endif // __LOCO_IR_DEPTHWISE_FILTER_SHAPE_H__
