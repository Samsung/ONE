/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
 */

#ifndef __ONERT_EXEC_FEATURE_READER_H__
#define __ONERT_EXEC_FEATURE_READER_H__

#include <cstdint>

namespace onert
{
namespace exec
{
namespace feature
{

/**
 * @brief  Class reads values of feature
 *         The interface class
 */
template <typename T> struct Reader
{
  /**
   * @brief  Destruct Reader object using default destructor
   */
  virtual ~Reader() = default;

  /**
   * @brief     Get the value used by three indexes
   * @param[in] ch  The depth index
   * @param[in] row The height index
   * @param[in] col The width index
   * @return    The value at the offset
   */
  virtual T at(uint32_t ch, uint32_t row, uint32_t col) const = 0;
  /**
   * @brief     Get the value used by four indexes
   * @param[in] batch The batch index
   * @param[in] ch    The depth index
   * @param[in] row   The height index
   * @param[in] col   The width index
   * @return    The value at the offset
   */
  virtual T at(uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) const = 0;
};

} // namespace feature
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_FEATURE_READER_H__
