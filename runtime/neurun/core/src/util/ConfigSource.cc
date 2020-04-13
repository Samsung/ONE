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

#include "util/ConfigSource.h"
#include "util/GeneralConfigSource.h"
#include "util/EnvConfigSource.h"

#include <array>
#include <algorithm>
#include <cassert>

#include "memory"

namespace neurun
{
namespace util
{

static std::unique_ptr<IConfigSource> _source;

void config_source(std::unique_ptr<IConfigSource> &&source) { _source = std::move(source); }

static IConfigSource *config_source()
{
  if (!_source)
  {
#ifdef ENVVAR_FOR_DEFAULT_CONFIG
    // Default ConfigSource is EnvConfigSource
    _source = std::make_unique<EnvConfigSource>();
#else
    _source = std::make_unique<GeneralConfigSource>();
#endif // ENVVAR_FOR_DEFAULT_CONFIG
  }
  return _source.get();
}

static std::string getConfigOrDefault(const std::string &key)
{
  static std::unordered_map<std::string, std::string> defaults;
  if (defaults.empty())
  {
#define CONFIG(Name, Type, Default)               \
  {                                               \
    auto name = std::string{#Name};               \
    defaults.emplace(name, std::string{Default}); \
  }

#include "util/Config.lst"

#undef CONFIG
  }

  // Treat empty string and absence of the value to be the same
  auto ret = config_source()->get(key);
  if (ret.empty())
  {
    auto itr = defaults.find(key);
    if (itr != defaults.end())
    {
      // Return the default value if exists
      ret = itr->second;
    }
  }

  return ret;
}

bool getConfigBool(const std::string &key)
{
  auto raw = getConfigOrDefault(key);
  static const std::array<std::string, 5> false_list{"0", "OFF", "FALSE", "N", "NO"};
  auto false_found = std::find(false_list.begin(), false_list.end(), raw);

  return (false_found == false_list.end());
}

int getConfigInt(const std::string &key)
{
  auto raw = getConfigOrDefault(key);
  return std::stoi(raw);
}

std::string getConfigString(const std::string &key) { return getConfigOrDefault(key); }

} // namespace util
} // namespace neurun

namespace neurun
{
namespace util
{
namespace config
{

#define CONFIG(Name, Type, Default) const char *Name = #Name;

#include "util/Config.lst"

#undef CONFIG

} // namespace config
} // namespace util
} // namespace neurun
