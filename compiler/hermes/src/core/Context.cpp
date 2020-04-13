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

#include "hermes/core/Context.h"

#include <cassert>

namespace hermes
{

const Config *Context::config(void) const
{
  // Return the current configuration
  return _config.get();
}

void Context::config(std::unique_ptr<Config> &&config)
{
  _config = std::move(config);

  // Apply updated configurations
  for (auto source : _sources)
  {
    source->reload(_config.get());
  }
}

void Context::post(std::unique_ptr<Message> &&msg)
{
  // Validate message
  assert((msg != nullptr) && "invalid message");
  assert((msg->text() != nullptr) && "missing text");

  // Take the ownership of a given message
  auto m = std::move(msg);

  // Notify appended sinks
  for (const auto &sink : _sinks)
  {
    sink->notify(m.get());
  }

  // TODO Stop the process if "FATAL" message is posted
}

void Context::attach(Source *source)
{
  // Configure source first
  source->reload(config());
  // Insert source
  _sources.insert(source);
}

void Context::detach(Source *source)
{
  // Remove source
  _sources.erase(source);
}

void Context::append(std::unique_ptr<Sink> &&sink)
{
  // Append sink
  _sinks.insert(std::move(sink));
}

} // namespace hermes
