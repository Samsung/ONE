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

#ifndef __HERMES_CONTEXT_H__
#define __HERMES_CONTEXT_H__

#include "hermes/core/Config.h"
#include "hermes/core/Source.h"
#include "hermes/core/Sink.h"
#include "hermes/core/MessageBus.h"

#include <memory>
#include <set>

namespace hermes
{

/**
 * @brief Logging controller
 *
 * This "Context" serves as a controller for associated logging source/sink.
 *
 * WARNING This "Context" is not yet thread-safe.
 * TODO Support multi-threaded application logging
 */
class Context final : private MessageBus, private Source::Registry, private Sink::Registry
{
public:
  /// @brief Get the global configuration
  const Config *config(void) const;
  /// @brief Update the global configuration
  void config(std::unique_ptr<Config> &&);

public:
  MessageBus *bus(void) { return this; }

private:
  /// This implements "post" method that MessageBus interface requires.
  void post(std::unique_ptr<Message> &&msg) override;

public:
  Source::Registry *sources(void) { return this; }

private:
  /// This implements "attach" method that "Source::Registry" interface requires.
  void attach(Source *source) override;
  /// This implements "detach" method that "Source::Registry" interface requires.
  void detach(Source *source) override;

public:
  Sink::Registry *sinks(void) { return this; }

private:
  /// This implements "append" method that "Sink::Registry" interface requires.
  void append(std::unique_ptr<Sink> &&sink) override;

private:
  std::unique_ptr<Config> _config;
  std::set<Source *> _sources;
  std::set<std::unique_ptr<Sink>> _sinks;
};

} // namespace hermes

#endif // __HERMES_CONTEXT_H__
