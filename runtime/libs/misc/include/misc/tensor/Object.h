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

/**
 * @file Object.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::Object class
 */

#ifndef __NNFW_MISC_TENSOR_OBJECT_H__
#define __NNFW_MISC_TENSOR_OBJECT_H__

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"
#include "misc/tensor/IndexIterator.h"
#include "misc/tensor/NonIncreasingStride.h"
#include "misc/tensor/Reader.h"

#include <vector>

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Class to build a tensor using specific generator
 * @tparam T  Type of tensor element
 */

template <typename T> class Object final : public Reader<T>
{
public:
  /**
   * @brief Function to generate tensor element
   */
  using Generator = std::function<T(const Shape &shape, const Index &index)>;

public:
  /**
   * @brief Construct a new @c Object object
   * @param[in] shape   Tensor shape
   * @param[in] fn      Function to generate tensor elements
   */
  Object(const Shape &shape, const Generator &fn) : _shape{shape}
  {
    // Set 'stride'
    _stride.init(shape);

    // Handle scalar object
    if (shape.rank() == 0)
    {
      _values.resize(1);
      _values.at(0) = fn(_shape, 0);
    }
    else
    {
      // Pre-allocate buffer
      _values.resize(_shape.dim(0) * _stride.at(0));

      // Set 'value'
      iterate(_shape) <<
        [this, &fn](const Index &index) { _values.at(_stride.offset(index)) = fn(_shape, index); };
    }
  }

public:
  /**
   * @brief Get reference of shape
   * @return Reference of shape
   */
  const Shape &shape(void) const { return _shape; }

public:
  /**
   * @brief Get and element of tensor
   * @param[in] index   Index of a tensor element
   * @return Value of tensor element
   */
  T at(const Index &index) const override { return _values.at(_stride.offset(index)); }

private:
  Shape _shape;
  NonIncreasingStride _stride;

private:
  std::vector<T> _values;
};

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_FEATURE_OBJECT_H__
