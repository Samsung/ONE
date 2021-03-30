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

#include "PConfigJson.h"

#include <gtest/gtest.h>

#include <sstream>

TEST(ConfigJsonTest, empty)
{
  std::stringstream ss;
  crew::JsonExport je(ss);

  je.open_brace();
  je.close_brace(true);

  ASSERT_TRUE(ss.str() == "{\n},\n");
}

TEST(ConfigJsonTest, keyvalue)
{
  std::stringstream ss;
  crew::JsonExport je(ss);

  je.open_brace("hello");
  je.key_val("key", "value", true);
  je.close_brace(true);

  ASSERT_TRUE(ss.str() == "\"hello\" : {\n  \"key\" : \"value\",\n},\n");
}

TEST(ConfigJsonTest, keyvaluearray)
{
  std::stringstream ss;
  crew::JsonExport je(ss);
  std::vector<std::string> vs = {"1", "2"};

  je.open_brace("hello");
  je.key_val("key", vs, true);
  je.close_brace(true);

  ASSERT_TRUE(ss.str() == "\"hello\" : {\n  \"key\" : [ \"1\", \"2\" ],\n},\n");
}

TEST(ConfigJsonTest, bracket)
{
  std::stringstream ss;
  crew::JsonExport je(ss);

  je.open_bracket("hello");
  je.close_bracket(true);

  ASSERT_TRUE(ss.str() == "\"hello\" : [\n],\n");
}
