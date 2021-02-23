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

#include "hermes/core/Source.h"

#include <memory>
#include <cassert>

namespace hermes
{

Source::Source()
{
  assert(_reg == nullptr);
  assert(_bus == nullptr);
}

Source::~Source()
{
  assert(_bus == nullptr);
  assert(_reg == nullptr);
}

void Source::activate(Registry *reg, MessageBus *bus)
{
  assert((_reg == nullptr) && (_bus == nullptr));

  _reg = reg;
  _bus = bus;

  _reg->attach(this);

  assert((_bus != nullptr) && (_reg != nullptr));
}

void Source::deactivate(void)
{
  assert((_bus != nullptr) && (_reg != nullptr));

  _reg->detach(this);

  _bus = nullptr;
  _reg = nullptr;

  assert((_reg == nullptr) && (_bus == nullptr));
}

void Source::reload(const Config *c) { c->configure(this, _setting); }

std::unique_ptr<MessageBuffer> Source::buffer(const Severity &) const
{
  // TODO Pass Severity
  return std::make_unique<MessageBuffer>(_bus);
}

} // namespace hermes
