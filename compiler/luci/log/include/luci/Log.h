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

#ifndef __LUCI_LOG_H__
#define __LUCI_LOG_H__

#include <hermes.h>

namespace luci
{

/**
 * @brief Logger Implementation
 */
class Logger final : public hermes::Source
{
public:
  Logger(hermes::Context *ctx);
  ~Logger();
};

/**
 * @brief Logger Configuration
 *
 * Users are able to turn logging on/off via LUCI_LOG environment variable.
 */
class LoggerConfig final : public hermes::Config
{
public:
  LoggerConfig();

public:
  void configure(const hermes::Source *, hermes::Source::Setting &) const final;
  void configure(const Logger *, hermes::Source::Setting &) const;

private:
  bool _enabled;
};

} // namespace luci

#include "luci/LoggingContext.h"

/**
 * HOW TO USE:
 *
 *   LOGGER(l);
 *
 *   INFO(l) << "Hello, World" << std::endl;
 *
 */
#define LOGGER(name) ::luci::Logger name{::luci::LoggingContext::get()};

// TODO Support FATAL, ERROR, WARN, and VERBOSE
#define INFO(name) HERMES_INFO(name)

// WARNING!
//
//   THE CURRENT IMPLEMENTATION IS NOT THREAD SAFE.
//

#endif // __LUCI_LOG_H__
