/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/ImporterEx.h"

#include <gtest/gtest.h>
#include <sstream>

TEST(ImporterEx, logs_to_stderr)
{
  std::ostringstream cerr_substitute;

  auto cerr_original_buffer = std::cerr.rdbuf();
  std::cerr.rdbuf(cerr_substitute.rdbuf());

  luci::ImporterEx importer;
  importer.importVerifyModule("/non/existent.path");

  std::cerr.rdbuf(cerr_original_buffer);

  ASSERT_GT(cerr_substitute.str().length(), 0);
}

TEST(ImporterEx, calls_external_error_handler)
{

  struct ErrorHandler
  {
    ErrorHandler(bool &flag) : _flag{flag} {}

    void operator()(const std::exception &) { _flag = true; }

    bool &_flag;
  };

  bool error_handler_called = false;
  ErrorHandler handler{error_handler_called};

  luci::ImporterEx importer{handler};
  importer.importVerifyModule("/non/existent.path");

  ASSERT_TRUE(error_handler_called);
}

TEST(ImporterEx, throws_with_empty_handler_N)
{
  std::function<void(const std::exception &)> empty_handler; // this doesn't contain any callable
  ASSERT_THROW(luci::ImporterEx{empty_handler}, std::runtime_error);
}
