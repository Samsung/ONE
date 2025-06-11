/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ARSER_PROMPT_H__
#define __ARSER_PROMPT_H__

#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace arser
{
namespace test
{

class Prompt
{
public:
  Prompt(const std::string &command)
  {
    std::istringstream iss(command);
    std::vector<std::string> token(std::istream_iterator<std::string>{iss},
                                   std::istream_iterator<std::string>());
    _arg = std::move(token);
    updateArgv();
  }
  int argc(void) const { return _argv.size(); }
  char **argv(void) { return _argv.data(); }

  // Manually set arguments from a list of strings.
  // This is for testing space-included strings.
  void set_arguments(const std::vector<std::string> &args)
  {
    _arg = args;
    updateArgv();
  }

  void set_arguments(std::initializer_list<std::string> args)
  {
    _arg = std::vector<std::string>(args);
    updateArgv();
  }

private:
  void updateArgv(void)
  {
    _argv.clear();
    _argv.reserve(_arg.size());
    for (const auto &t : _arg)
    {
      _argv.push_back(const_cast<char *>(t.data()));
    }
  }

private:
  std::vector<char *> _argv;
  std::vector<std::string> _arg;
};

} // namespace test
} // namespace arser

#endif // __ARSER_PROMPT_H__
