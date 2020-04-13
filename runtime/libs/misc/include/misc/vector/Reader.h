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
 * @file     Reader.h
 * @brief    This file contains Reader class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_VECTOR_READER_H__
#define __NNFW_MISC_VECTOR_READER_H__

#include <cstdint>

namespace nnfw
{
namespace misc
{
namespace vector
{

/**
 * @brief  Class reads values of vector
 *         The interface class
 */
template <typename T> struct Reader
{
  /**
   * @brief  Destruct Reader object using default destructor
   */
  virtual ~Reader() = default;

  /**
   * @brief   Get the value used by the index
   * @param[in]  nth    The vector index
   * @return  The value at the offset
   */
  virtual T at(uint32_t nth) const = 0;
};

} // namespace vector
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_VECTOR_READER_H__
