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
 * @file logging.h
 * @brief This file contains Context class for logging.
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __PURE_ARM_COMPUTE_LOGGING_H__
#define __PURE_ARM_COMPUTE_LOGGING_H__

#include <iostream>

namespace logging
{

/**
 * @brief class to define Context for logging
 */
class Context
{
public:
  /**
   * @brief Construct default
   */
  Context() : _enabled{false}
  {
    auto env = std::getenv("PURE_ARM_COMPUTE_LOG_ENABLE");

    if (env && std::strtol(env, NULL, 0) > 0)
    {
      _enabled = true;
    }
  }

public:
  /**
   * @brief Get @c true if PURE_ARM_COMPUTE_LOG_ENABLE has been set as environment value, otherwise
   * @c false
   * @return @c true if PURE_ARM_COMPUTE_LOG_ENABLE has been set as environment value, otherwise @c
   * false
   */
  bool enabled(void) const { return _enabled; }

private:
  bool _enabled;
};

/**
 * @brief static Context class for logging
 */
static Context ctx;

} // namespace logging

#define VERBOSE(name)           \
  if (::logging::ctx.enabled()) \
  std::cout << "[" << #name << "] "

#endif // __PURE_ARM_COMPUTE_LOGGING_H__
