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
 * @file     TextFormatter.h
 * @brief    This file contains TextFormatter class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_FEATURE_TEXT_FORMATTER_H__
#define __NNFW_MISC_FEATURE_TEXT_FORMATTER_H__

#include "misc/feature/Shape.h"
#include "misc/feature/Reader.h"

#include <ostream>
#include <iomanip>
#include <limits>

namespace nnfw
{
namespace misc
{
namespace feature
{

/**
 * @brief   Class to print operand of feature to ostream in the given string format
 */
template <typename T> class TextFormatter
{
public:
  /**
   * @brief  Construct TextFormatter object with an operand's information.
   * @param[in]  shape  The shape of an operand
   * @param[in]  data   The data of an operand
   */
  TextFormatter(const Shape &shape, const Reader<T> &data) : _shape(shape), _data(data)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief   Get Shape of feature as the lvalue reference
   * @return  Shape of feature
   */
  const Shape &shape(void) const { return _shape; }
  /**
   * @brief   Get Reader<T> that can read the data of an operand
   * @return  Reader<T>
   */
  const Reader<T> &data(void) const { return _data; }

private:
  /**
   * @brief   Shape of feature
   */
  const Shape &_shape;
  /**
   * @brief   Reader<T> that can read the data of an operand
   */
  const Reader<T> &_data;
};

/**
 * @brief   Print operand of feature
 * @param[in]   os   Standard output stream
 * @param[in]   fmt  TextFormatter to print information of an operand
 * @return  Standard output stream
 */
template <typename T> std::ostream &operator<<(std::ostream &os, const TextFormatter<T> &fmt)
{
  const auto &shape = fmt.shape();

  for (uint32_t ch = 0; ch < shape.C; ++ch)
  {
    os << "  Channel " << ch << ":" << std::endl;
    for (uint32_t row = 0; row < shape.H; ++row)
    {
      os << "    ";
      for (uint32_t col = 0; col < shape.W; ++col)
      {
        const auto value = fmt.data().at(ch, row, col);
        os << std::right;
        os << std::fixed;
        os << std::setw(std::numeric_limits<T>::digits10 + 2);
        os << std::setprecision(5);
        os << value;
        os << " ";
      }
      os << std::endl;
    }
  }

  return os;
}

} // namespace feature
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_FEATURE_TEXT_FORMATTER_H__
