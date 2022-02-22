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

#ifndef __HERMES_MESSAGE_BUFFER_H__
#define __HERMES_MESSAGE_BUFFER_H__

#include "hermes/core/MessageBus.h"
#include "hermes/core/Severity.h"

#include <ostream>
#include <sstream>

namespace hermes
{

/**
 * @brief A buffer for a message under construction
 *
 * MessageBuffer will post the buffered message on destruction.
 */
class MessageBuffer final
{
public:
  MessageBuffer(MessageBus *);
  MessageBuffer(MessageBus *bus, SeverityCategory severity);
  ~MessageBuffer();

public:
  std::ostream &os(void) { return _ss; }

private:
  MessageBus *_bus;
  SeverityCategory _severity = SeverityCategory::INFO;

  /// @brief Content buffer
  std::stringstream _ss;
};

} // namespace hermes

#endif // __HERMES_MESSAGE_BUFFER_H__
