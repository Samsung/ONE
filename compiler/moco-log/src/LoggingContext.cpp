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

#include "moco/LoggingContext.h"
#include "moco/Log.h"

#include <hermes/ConsoleReporter.h>

#include <memory>

namespace moco
{

hermes::Context *LoggingContext::get(void)
{
  static hermes::Context *ctx = nullptr;

  if (ctx == nullptr)
  {
    ctx = new hermes::Context;
    ctx->sinks()->append(std::make_unique<hermes::ConsoleReporter>());
    ctx->config(std::make_unique<LoggerConfig>());
  }

  return ctx;
}

} // namespace moco
