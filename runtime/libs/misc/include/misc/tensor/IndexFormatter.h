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
 * @file IndexFormatter.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::IndexFormatter class
 */

#ifndef __NNFW_MISC_TENSOR_INDEX_FORMATTER_H__
#define __NNFW_MISC_TENSOR_INDEX_FORMATTER_H__

#include "misc/tensor/Index.h"

#include <ostream>

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Class to send @c Index object to output stream
 */
class IndexFormatter
{
public:
  /**
   * @brief Construct a new @c IndexFormatter object
   * @param[in] index   index to be sent to output stream
   */
  IndexFormatter(const nnfw::misc::tensor::Index &index) : _index(index)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get an @c Index object
   * @return @c Index object previously passed to the constructor
   */
  const nnfw::misc::tensor::Index &index(void) const { return _index; }

private:
  const nnfw::misc::tensor::Index &_index;
};

/**
 * @brief Send @c IndexFormatter object to output stream
 * @param[in] os    Output stream
 * @param[in] fmt   @c IndexFormatter object that is sent to output stream
 * @return Output stream
 */
std::ostream &operator<<(std::ostream &os, const IndexFormatter &fmt);

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_INDEX_FORMATTER_H__
