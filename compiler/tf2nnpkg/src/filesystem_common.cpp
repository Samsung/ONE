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

#include "filesystem.h"

namespace filesystem
{

std::string join(const std::string &path1, const std::string &path2)
{
  // TODO check path1 does not end with separator
  // TODO check path2 does not start with separator
  return path1 + separator() + path2;
}

std::string basename(const std::string &path)
{
  auto last_index = path.find_last_of(separator());

  // No separator
  if (last_index == std::string::npos)
    return path;

  // Trailing separator
  if (last_index + separator().size() == path.size())
    return basename(path.substr(0, last_index));

  return path.substr(last_index + separator().size());
}

} // namespace filesystem
