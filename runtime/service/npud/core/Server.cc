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
#include "util/Logging.h"

#include <thread>

namespace npud
{
namespace core
{

std::atomic_bool Server::_isRunning(false);

Server::Server() noexcept
  : _mainloop(g_main_loop_new(NULL, FALSE), g_main_loop_unref), _signal(std::make_unique<Signal>()),
    _core(std::make_unique<Core>()), _dbus(std::make_unique<DBus>())
{
}

bool Server::isServiceReady()
{
  if (!_isRunning.load())
  {
    VERBOSE(Server) << "Server is not started." << std::endl;
    return false;
  }

  if (!_dbus->isReady())
  {
    VERBOSE(Server) << "DBus service is not ready." << std::endl;
    return false;
  }

  return true;
}

void Server::run(void)
{
  VERBOSE(Server) << "Starting Server\n";

  if (_isRunning.exchange(true))
  {
    return;
  }

  _core->init();

  g_main_loop_run(_mainloop.get());
}

void Server::stop(void)
{
  VERBOSE(Server) << "Stop Server\n";

  if (!_isRunning.load())
  {
    return;
  }

  while (!g_main_loop_is_running(_mainloop.get()))
  {
    std::this_thread::yield();
  }

  _core->deinit();

  g_main_loop_quit(_mainloop.get());
  _isRunning = false;
}

} // namespace core
} // namespace npud
