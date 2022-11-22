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

#include "Signal.h"
#include "Server.h"
#include "util/Logging.h"

#include <csignal>

namespace npud
{
namespace core
{

Signal::Signal(void) noexcept { init(); }

void Signal::init(void)
{
  // NOTE Types of signals
  // SIGTERM: termination request, sent to the program
  // SIGINT:  external interrupt, usually initiated by the user
  // SIGILL:	invalid program image, such as invalid instruction
  // SIGABRT:	abnormal termination condition, as is e.g. initiated by std::abort()
  // SIGFPE: 	erroneous arithmetic operation such as divide by zero
  // from https://en.cppreference.com/w/cpp/utility/program/SIG_types
  std::signal(SIGTERM, handleSignal);
  std::signal(SIGINT, handleSignal);
  std::signal(SIGILL, handleSignal);
  std::signal(SIGABRT, handleSignal);
  std::signal(SIGFPE, handleSignal);
}

void Signal::handleSignal(int signum)
{
  VERBOSE(signal) << "Signal received: " << strsignal(signum) << "(" << signum << ")\n";
  Server::instance().stop();
}

} // namespace core
} // namespace npud
