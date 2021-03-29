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

#include "crew/PConfigIni.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

TEST(ConfigIniTest, read_ini_non_exist_file)
{
  EXPECT_THROW(crew::read_ini("/hello/world/not_a_file"), std::runtime_error);
}

TEST(ConfigIniTest, read_ini_simple)
{
  std::stringstream ss;

  ss << "[hello]\nkey=world\n";

  auto str = ss.str();
  auto sections = crew::read_ini(str.c_str(), str.length());
  ASSERT_EQ(1UL, sections.size());

  auto its = sections.begin();
  ASSERT_NE(sections.end(), its);
  EXPECT_TRUE("hello" == its->name);
  ASSERT_EQ(1UL, its->items.size());

  auto it = its->items.begin();
  ASSERT_NE(its->items.end(), it);
  EXPECT_TRUE("key" == it->first);
  EXPECT_TRUE("world" == it->second);
}

TEST(ConfigIniTest, read_ini_simple_N)
{
  std::stringstream ss;

  ss << "key=value\nhello=world\n";

  auto str = ss.str();

  EXPECT_THROW(crew::read_ini(str.c_str(), str.length()), std::runtime_error);
}

TEST(ConfigIniTest, read_ini_comment)
{
  std::stringstream ss;

  ss << "[hello]\n;comment=skip\n#comment=skip\nkey=world\n";

  auto str = ss.str();
  auto sections = crew::read_ini(str.c_str(), str.length());
  ASSERT_EQ(1UL, sections.size());

  auto its = sections.begin();
  ASSERT_NE(sections.end(), its);
  EXPECT_TRUE("hello" == its->name);
  ASSERT_EQ(1UL, its->items.size());

  auto it = its->items.begin();
  ASSERT_NE(its->items.end(), it);
  EXPECT_TRUE("key" == it->first);
  EXPECT_TRUE("world" == it->second);
}
