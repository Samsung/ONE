/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Log.h"

#include <hermes/ConsoleReporter.h>

#include <cstdlib>
#include <iostream>

// TODO Extract these lexical conversion routines as a library
namespace
{

/**
 * @brief Convert C-string as a value of type T
 *
 * safecast(s, v) returns v if s is nullptr.
 */
template <typename T> T safecast(const char *, const T &);

template <> bool safecast<bool>(const char *s, const bool &value)
{
  return (s == nullptr) ? value : (std::stoi(s) != 0);
}

} // namespace

namespace exo
{

//
// Logger
//
Logger::Logger(hermes::Context *ctx) { activate(ctx->sources(), ctx->bus()); }
Logger::~Logger() { deactivate(); }

//
// LoggerConfig
//
LoggerConfig::LoggerConfig()
{
  // Turn on logging if EXO_LOG is set as non-zero value
  _enabled = safecast<bool>(std::getenv("EXO_LOG"), false);
}

void LoggerConfig::configure(const hermes::Source *source, hermes::Source::Setting &setting) const
{
  // Let's ignore hermes::Sources if that is not a exo logger
  if (auto logger = dynamic_cast<const Logger *>(source))
  {
    configure(logger, setting);
  }
}

void LoggerConfig::configure(const Logger *, hermes::Source::Setting &setting) const
{
  if (_enabled)
  {
    // Enable all catagories
    setting.accept_all();
  }
  else
  {
    // Disable all catagories
    setting.reject_all();
  }
}

} // namespace exo
