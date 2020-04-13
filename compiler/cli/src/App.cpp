/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "cli/App.h"

#include <iostream>
#include <cassert>

namespace cli
{

App::App(const std::string &name) : _name{name}
{
  // DO NOTHING
}

App &App::insert(const std::string &tag, std::unique_ptr<Command> &&command)
{
  assert(_commands.find(tag) == _commands.end());

  _commands[tag] = std::move(command);

  return (*this);
}

int App::run(int argc, const char *const *argv) const
{
  if (argc < 1)
  {
    std::cerr << "ERROR: COMMAND is not provided" << std::endl;
    usage(std::cerr);
    return 255;
  }

  const std::string command{argv[0]};

  auto it = _commands.find(command);

  if (it == _commands.end())
  {
    std::cerr << "ERROR: '" << command << "' is not a valid command" << std::endl;
    usage(std::cerr);
    return 255;
  }

  return it->second->run(argc - 1, argv + 1);
}

void App::usage(std::ostream &os) const
{
  os << std::endl;
  os << "USAGE: " << _name << " [COMMAND] ..." << std::endl;
  os << std::endl;
  os << "SUPPORTED COMMANDS:" << std::endl;
  for (auto it = _commands.begin(); it != _commands.end(); ++it)
  {
    os << "  " << it->first << std::endl;
  }
}

} // namespace cli
