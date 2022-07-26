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

#include <thread>
#include <iostream>

namespace npud
{
namespace core
{

std::atomic_bool Server::_isRunning(false);

Server::Server() noexcept
  : _mainloop(g_main_loop_new(NULL, FALSE), g_main_loop_unref), _signal(std::make_unique<Signal>())
{
}

void Server::run(void)
{
  std::cout << "Starting Server" << std::endl;

  if (_isRunning.exchange(true))
  {
    throw std::runtime_error("Mainloop is already running.");
  }

  g_main_loop_run(_mainloop.get());
}

void Server::stop(void)
{
  std::cout << "Stop Server" << std::endl;

  if (!_isRunning.load())
  {
    throw std::runtime_error("Mainloop is not running");
  }

  while (!g_main_loop_is_running(_mainloop.get()))
  {
    std::this_thread::yield();
  }

  g_main_loop_quit(_mainloop.get());
  _isRunning = false;
}

} // namespace core
} // namespace npud
