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

#ifndef __NNFW_MISC_VECTOR_OBJECT_H__
#define __NNFW_MISC_VECTOR_OBJECT_H__

#include "misc/vector/Reader.h"

#include <vector>
#include <functional>

namespace nnfw
{
namespace misc
{
namespace vector
{

/**
 * @brief  Class to have information of the operand for vector
 */
template <typename T> class Object final : public Reader<T>
{
public:
  using Generator = std::function<T(int32_t size, int32_t offset)>;

public:
  /**
   * @brief  Construct Object object with size of vector and set value used by Generator
   * @param[in]  size    The size of vector
   * @param[in]  gen     A function to set values of operand tensor
   */
  Object(int32_t size, const Generator &gen) : _size{size}
  {
    _value.resize(_size);

    for (int32_t offset = 0; offset < size; ++offset)
    {
      _value.at(offset) = gen(size, offset);
    }
  }

public:
  /**
   * @brief   Get size of vector
   * @return  Size of vector
   */
  int32_t size(void) const { return _size; }

public:
  /**
   * @brief   Get the value used by index
   * @param[in]  nth    The vector index
   * @return  The value at the offset
   */
  T at(uint32_t nth) const override { return _value.at(nth); }

private:
  /**
   * @brief   Size of vector
   */
  const int32_t _size;
  /**
   * @brief   The tensor vector of operand
   */
  std::vector<T> _value;
};

} // namespace vector
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_VECTOR_OBJECT_H__
