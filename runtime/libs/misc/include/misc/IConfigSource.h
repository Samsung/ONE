/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_MISC_I_CONFIG_SOURCE_H__
#define __NNFW_MISC_I_CONFIG_SOURCE_H__

#include <string>

namespace nnfw
{
namespace misc
{

struct IConfigSource
{
  /**
   * @brief Destroy the IConfigSource object
   */
  virtual ~IConfigSource() = default;

  /**
   * @brief get the value for the matching key
   *
   * @param key string key to search
   * @return string value associated with the key
   */
  virtual std::string get(const std::string &key) const = 0;
};

} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_I_CONFIG_SOURCE_H__
