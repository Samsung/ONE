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
 * @file     Object.h
 * @brief    This file contains Object class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_FEATURE_OBJECT_H__
#define __NNFW_MISC_FEATURE_OBJECT_H__

#include "misc/feature/Shape.h"
#include "misc/feature/Index.h"
#include "misc/feature/Reader.h"

#include <vector>

namespace nnfw
{
namespace misc
{
namespace feature
{

/**
 * @brief  Class to have information of the operand for feature
 */
template <typename T> class Object final : public Reader<T>
{
public:
  using Generator = std::function<T(const Shape &shape, const Index &index)>;

public:
  /**
   * @brief  Construct Object object with Shape of feature and set value used by Generator
   * @param[in]  shape   Reference of Shape for feature
   * @param[in]  fn      A function to set values of operand tensor
   */
  Object(const Shape &shape, const Generator &fn) : _shape{shape}
  {
    _value.resize(_shape.C * _shape.H * _shape.W);

    for (int32_t ch = 0; ch < _shape.C; ++ch)
    {
      for (int32_t row = 0; row < _shape.H; ++row)
      {
        for (int32_t col = 0; col < _shape.W; ++col)
        {
          _value.at(offsetOf(ch, row, col)) = fn(_shape, Index{ch, row, col});
        }
      }
    }
  }

public:
  /**
   * @brief   Get Shape of feature as the reference
   * @return  The reference of the width value
   */
  const Shape &shape(void) const { return _shape; }

public:
  /**
   * @brief   Get the value used by three indexes
   * @param[in]   ch   The depth index
   * @param[in]   row  The height index
   * @param[in]   col  The width index
   * @return  The value at the offset
   */
  T at(uint32_t ch, uint32_t row, uint32_t col) const override
  {
    return _value.at(offsetOf(ch, row, col));
  }

private:
  /**
   * @brief   Get the offset value at three indexes
   * @param[in]   ch   The depth index
   * @param[in]   row  The height index
   * @param[in]   col  The width index
   * @return  The offset value
   */
  uint32_t offsetOf(uint32_t ch, uint32_t row, uint32_t col) const
  {
    return ch * _shape.H * _shape.W + row * _shape.W + col;
  }

private:
  /**
   * @brief   Shape of operand
   */
  Shape _shape;
  /**
   * @brief   The tensor vector of operand
   */
  std::vector<T> _value;
};

} // namespace feature
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_FEATURE_OBJECT_H__
