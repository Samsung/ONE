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

#ifndef __ONE_SERVICE_NPUD_UTIL_LOGGING_H__
#define __ONE_SERVICE_NPUD_UTIL_LOGGING_H__

#include <iostream>
#include <cstring>

#include "ConfigSource.h"

namespace npud
{
namespace util
{
namespace logging
{
class Context
{
public:
  Context() noexcept : _enabled{false}
  {
    const auto env = util::getConfigBool(util::config::NPUD_LOG_ENABLE);

    if (env)
    {
      _enabled = true;
    }
  }

  static Context &get() noexcept
  {
    static Context ctx;
    return ctx;
  }

public:
  bool enabled(void) const { return _enabled; }

private:
  bool _enabled;
};

static Context &ctx = Context::get();

inline std::string decorated_name(const char *input)
{
  const int min_prefix = 16;
  std::string prefix(input);
  auto len_prefix = prefix.size();
  if (len_prefix > min_prefix)
    return "[" + prefix + "] ";
  std::string spaces((min_prefix - len_prefix) / 2, ' ');
  return (len_prefix % 2 ? "[ " : "[") + spaces + prefix + spaces + "] ";
}
} // namespace logging
} // namespace util
} // namespace npud

#define VERBOSE(name)                       \
  if (::npud::util::logging::ctx.enabled()) \
  std::cout << ::npud::util::logging::decorated_name(#name)

#define VERBOSE_F()                         \
  if (::npud::util::logging::ctx.enabled()) \
  std::cout << ::npud::util::logging::decorated_name(__func__)

#define WHEN_LOG_ENABLED(METHOD)            \
  if (::npud::util::logging::ctx.enabled()) \
    do                                      \
    {                                       \
      METHOD;                               \
  } while (0)

#endif // __ONE_SERVICE_NPUD_UTIL_LOGGING_H__
