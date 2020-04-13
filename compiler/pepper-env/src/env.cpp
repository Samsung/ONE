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

#include "pepper/env.h"

//
// KVStoreInterface
//
namespace pepper
{

std::string KVStoreInterface<KVStoreTrait::Queryable>::get(const std::string &key,
                                                           const std::string &default_value) const
{
  if (auto p = query(key.c_str()))
  {
    return p;
  }
  return default_value;
}

} // namespace pepper

//
// ProcessEnvironment
//
#include <cstdlib>

namespace pepper
{

const char *ProcessEnvironment::query(const char *k) const { return std::getenv(k); }

} // namespace pepper
