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

#include "Server.h"

#include <util/Logging.h>

namespace npud
{
namespace core
{

EventLoop Server::_loop;

void Server::run(void)
{
  VERBOSE(Server) << "Starting Server\n";

  if (_loop.is_running())
  {
    VERBOSE(Server) << "Event loop is running\n";
    return;
  }

  _loop.run();
}

void Server::stop(void)
{
  VERBOSE(Server) << "Stop Server\n";

  if (!_loop.is_running())
  {
    VERBOSE(Server) << "Event loop is not running\n";
    return;
  }

  _loop.stop();
}

} // namespace core
} // namespace npud
