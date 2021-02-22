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

#include <memory>

#include <gtest/gtest.h>

class RecordCommand final : public cli::Command
{
public:
  RecordCommand(int ret, std::string &out) : _ret{ret}, _out(out)
  {
    // DO NOTHING
  }

public:
  int run(int argc, const char *const *argv) const override
  {
    _out += std::to_string(argc);

    for (int n = 0; n < argc; ++n)
    {
      _out += ";";
      _out += argv[n];
    }

    return _ret;
  }

private:
  int const _ret;
  std::string &_out;
};

TEST(APP, run)
{
  cli::App app("test");

  std::string args;
  app.insert("record", std::make_unique<RecordCommand>(3, args));

  const char *argv[] = {"record", "hello", "world"};

  int ret = app.run(3, argv);

  ASSERT_EQ(ret, 3);
  ASSERT_EQ(args, "2;hello;world");
}
