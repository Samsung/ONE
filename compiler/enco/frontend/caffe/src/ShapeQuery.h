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

#ifndef __SHAPE_QUERY_H__
#define __SHAPE_QUERY_H__

#include <nncc/core/ADT/tensor/Shape.h>

/**
 * @brief A wrapper class for an integer number that specifies axis
 *
 * Several Caffe layers includes 'axis' parameter (which may be negative) which specifies
 * some axis required for operation.
 *
 * Here are several examples:
 * - Convolution layer uses 'axis' parameter to specify "channel" axis
 *   (http://caffe.berkeleyvision.org/tutorial/layers/convolution.html)
 * - Concat layer uses 'axis' parameter to specify axis to be concatenated
 *   (http://caffe.berkeleyvision.org/tutorial/layers/concat.html)
 *
 * AxisSpecifier class is introduced to distinguish this 'axis' parameter from other integers
 * (to prevent possible mistake).
 */
class AxisSpecifier
{
public:
  explicit AxisSpecifier(int32_t value) : _value{value}
  {
    // DO NOTHING
  }

public:
  int32_t value(void) const { return _value; }

private:
  int32_t _value = 1;
};

AxisSpecifier axis_specifier(int32_t value);

/**
 * @brief A wrapper class that allows additional queries over tensor shape.
 */
class ShapeQuery
{
public:
  explicit ShapeQuery(const nncc::core::ADT::tensor::Shape *shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  /// @brief Return the dimension number (axis) specified by a given axis specifier
  uint32_t axis(const AxisSpecifier &) const;

private:
  const nncc::core::ADT::tensor::Shape *_shape;
};

ShapeQuery query_on(const nncc::core::ADT::tensor::Shape &);

#endif // __SHAPE_QUERY_H__
