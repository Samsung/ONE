/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_UTIL_LOGGING_H__
#define __ONERT_UTIL_LOGGING_H__

#include <iostream>

#include "util/ConfigSource.h"

namespace onert
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
    const auto env = util::getConfigBool(util::config::ONERT_LOG_ENABLE);

    if (env)
    {
      _enabled = true;
    }
  }

  static Context &get() noexcept;

public:
  bool enabled(void) const { return _enabled; }

private:
  bool _enabled;
};

static Context &ctx = Context::get();

} // namespace logging
} // namespace util
} // namespace onert

#define VERBOSE(name)                        \
  if (::onert::util::logging::ctx.enabled()) \
  std::cout << "[" << #name << "] "

#define VERBOSE_F()                          \
  if (::onert::util::logging::ctx.enabled()) \
  std::cout << "[" << __func__ << "] "

#define WHEN_LOG_ENABLED(METHOD)             \
  if (::onert::util::logging::ctx.enabled()) \
    do                                       \
    {                                        \
      METHOD;                                \
  } while (0)

#endif // __ONERT_UTIL_LOGGING_H__
