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

#include <gtest/gtest.h>

#include "arser/arser.h"

#include "Prompt.h"

using namespace arser;

/**
 * [WARNING] DO NOT GIVE THE ARSER '-h' or '--help' OPTION IN BELOW TESTS.
 *
 * arser exits with code 0 when '-h' option is given, which forces googletest to pass.
 */

TEST(HelpMessageTest, Default)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--dummy").nargs(0).help("Dummy optional argument");

  std::ostringstream oss;
  std::string expected_out = "Usage: ./arser [-h] [--dummy] \n"
                             "\n"
                             "[Optional argument]\n"
                             "-h, --help	Show help message and exit\n"
                             "--dummy   \tDummy optional argument\n";

  test::Prompt prompt("./arser --dummy");
  /* act */
  arser.parse(prompt.argc(), prompt.argv());
  oss << arser;

  /* assert */
  EXPECT_EQ(expected_out, oss.str());
}

TEST(HelpMessageTest, ShortOption)
{
  /* arrange */
  Arser arser;

  arser.add_argument("-v", "--verbose").nargs(0).help("Provides additional details");

  std::ostringstream oss;
  std::string expected_out = "Usage: ./arser [-h] [-v] \n"
                             "\n"
                             "[Optional argument]\n"
                             "-h, --help   \tShow help message and exit\n"
                             "-v, --verbose\tProvides additional details\n";

  test::Prompt prompt("./arser -v");
  /* act */
  arser.parse(prompt.argc(), prompt.argv());
  oss << arser;

  /* assert */
  EXPECT_EQ(expected_out, oss.str());
}
