/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "EnvConfigSource.h"

#include <cstdlib>

namespace npud
{
namespace util
{

std::string EnvConfigSource::get(const std::string &key) const
{
  const char *value = std::getenv(key.c_str());
  if (value != nullptr)
  {
    return value;
  }
  else
  {
    return GeneralConfigSource::get(key);
  }
}

} // namespace util
} // namespace npud
