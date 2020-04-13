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

#ifndef __CLI_FUNCTION_COMMAND_H__
#define __CLI_FUNCTION_COMMAND_H__

#include <cli/Command.h>

namespace cli
{

class FunctionCommand final : public Command
{
public:
  // NOTE The use of pure funtion pointer here is intended to disallow variable capture
  using Entry = int (*)(int argc, const char *const *argv);

public:
  FunctionCommand(const Entry &entry) : _entry{entry}
  {
    // DO NOTHING
  }

public:
  int run(int argc, const char *const *argv) const override { return _entry(argc, argv); };

private:
  Entry const _entry;
};

} // namespace cli

#endif // __CLI_FUNCTION_COMMAND_H__
