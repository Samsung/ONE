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

#include <vconone/vconone.h>

#include <gtest/gtest.h>

TEST(vconone, version_number)
{
  auto v = vconone::get_number();

  ASSERT_NE(0x0000000000000000ULL, v.v);
}

TEST(vconone, version_string)
{
  auto str = vconone::get_string();

  ASSERT_NE("..", str);
  ASSERT_NE("", str);
}

TEST(vconone, version_string4)
{
  auto str = vconone::get_string4();

  ASSERT_NE("...", str);
  ASSERT_NE("", str);
}

TEST(vconone, copyright)
{
  auto str = vconone::get_copyright();

  ASSERT_NE("", str);
}
