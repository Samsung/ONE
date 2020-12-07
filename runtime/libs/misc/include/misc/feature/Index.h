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
 * @file     Index.h
 * @brief    This file contains Index class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_FEATURE_INDEX_H__
#define __NNFW_MISC_FEATURE_INDEX_H__

#include <cstdint>

namespace nnfw
{
namespace misc
{
namespace feature
{

/**
 * @brief  Class to have the index information for calculating the offset.
 */
class Index
{
public:
  /**
   * @brief  Construct Index object using default constrcutor
   */
  Index() = default;

public:
  /**
   * @brief  Construct Index object with three indexes of dimensions
   * @param[in]  ch    The depth index
   * @param[in]  row   The heigth index
   * @param[in]  col   The width index
   */
  Index(int32_t ch, int32_t row, int32_t col) : _batch{1}, _ch{ch}, _row{row}, _col{col}
  {
    // DO NOTHING
  }
  /**
   * @brief  Construct Index object with four indexes of dimensions
   * @param[in]  batch The batch index
   * @param[in]  ch    The depth index
   * @param[in]  row   The height index
   * @param[in]  col   The width index
   */
  Index(int32_t batch, int32_t ch, int32_t row, int32_t col)
    : _batch{batch}, _ch{ch}, _row{row}, _col{col}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief   Get the batch index
   * @return  The batch index
   */
  int32_t batch(void) const { return _batch; }
  /**
   * @brief   Get the depth index
   * @return  The depth index
   */
  int32_t ch(void) const { return _ch; }
  /**
   * @brief   Get the height index
   * @return  The height index
   */
  int32_t row(void) const { return _row; }
  /**
   * @brief   Get the width index
   * @return  The width index
   */
  int32_t col(void) const { return _col; }

public:
  /**
   * @brief   Get the batch index as the lvalue reference
   * @return  The reference of the batch value
   */
  int32_t &batch(void) { return _batch; }
  /**
   * @brief   Get the depth index as the lvalue reference
   * @return  The reference of the depth value
   */
  int32_t &ch(void) { return _ch; }
  /**
   * @brief   Get the height index as the lvalue reference
   * @return  The reference of the height value
   */
  int32_t &row(void) { return _row; }
  /**
   * @brief   Get the width index as the lvalue reference
   * @return  The reference of the width value
   */
  int32_t &col(void) { return _col; }

private:
  /**
   * @brief  The batch index
   */
  int32_t _batch;
  /**
   * @brief  The depth index
   */
  int32_t _ch;
  /**
   * @brief  The height index
   */
  int32_t _row;
  /**
   * @brief  The width index
   */
  int32_t _col;
};

} // namespace feature
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_FEATURE_INDEX_H__
