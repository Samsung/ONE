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
 * @file Reader.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::Reader struct
 */

#ifndef __NNFW_MISC_TENSOR_READER_H__
#define __NNFW_MISC_TENSOR_READER_H__

#include "misc/tensor/Index.h"
#include "misc/tensor/Shape.h"

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief     Struct to read element of tensor
 * @tparam T  Type of elements in tensor
 */
template <typename T> struct Reader
{
  /**
   * @brief Destroy the Reader object
   */
  virtual ~Reader() = default;

  /**
   * @brief Get an element of tensor
   * @param[in] index   Index specifying indexes of tensor element
   * @return The value of specificed element
   */
  virtual T at(const Index &index) const = 0;

  /**
   * @brief   Get shape of tensor
   * @return  Reference of shape
   */
  virtual const Shape &shape(void) const = 0;
};

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_READER_H__
