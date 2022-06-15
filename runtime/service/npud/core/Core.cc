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

#include "Core.h"

#include <iostream>

EventLoop Core::_loop;

void Core::run(void)
{
  std::cout << "Starting Core" << std::endl;

  if (_loop.is_running())
  {
    std::cerr << "event loop is running" << std::endl;
    return;
  }

  _loop.run();
}

void Core::stop(void)
{
  std::cout << "Stop Core" << std::endl;

  if (!_loop.is_running())
  {
    std::cerr << "event loop is not running" << std::endl;
    return;
  }

  _loop.stop();
}
