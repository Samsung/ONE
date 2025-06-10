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

#ifndef __ONERT_UTIL_CONFIG_SOURCE_H__
#define __ONERT_UTIL_CONFIG_SOURCE_H__

#include <string>
#include <unordered_map>

namespace onert::util
{

// TODO: Apply Heterogeneous lookup for unordered containers (transparent hashing) since C++20
//       to use `std::string_view` with lookup functions in unordered containers
using CfgKeyValues = std::unordered_map<std::string, std::string>;

void setConfigKeyValues(const CfgKeyValues &keyValues);

bool toBool(const std::string &val);
int toInt(const std::string &val);

bool getConfigBool(const std::string &key);
int getConfigInt(const std::string &key);
std::string getConfigString(const std::string &key);

} // namespace onert::util

namespace onert::util::config
{

#define CONFIG(Name, Type, Default) extern const char *Name;

#include "Config.lst"

#undef CONFIG

} // namespace onert::util::config

#endif // __ONERT_UTIL_CONFIG_SOURCE_H__
