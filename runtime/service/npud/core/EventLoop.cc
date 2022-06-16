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

#include "EventLoop.h"

#include <iostream>
#include <cassert>

namespace npud
{
namespace core
{
EventLoop::EventLoop() : _mainloop(NULL), _running(false)
{
  std::cout << "EventLoop: constructor" << std::endl;
  _mainloop = g_main_loop_new(NULL, FALSE);
  assert(_mainloop);
}

EventLoop::~EventLoop() { std::cout << "EventLoop: destructor" << std::endl; }

void EventLoop::run(void)
{
  std::cout << "EventLoop: run" << std::endl;

  _running.store(true);
  g_main_loop_run(_mainloop);
}

void EventLoop::stop(void)
{
  std::cout << "EventLoop: stop" << std::endl;

  if (!_running.load())
  {
    return;
  }

  g_main_loop_quit(_mainloop);
  g_main_loop_unref(_mainloop);
  _running.store(false);
}

bool EventLoop::is_running(void) { return _running.load(); }
} // namespace core
} // namespace npud
