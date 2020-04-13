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

#include "pp/Format.h"

#include <gtest/gtest.h>

TEST(FORMAT, simple_string)
{
  ASSERT_EQ(pp::fmt("Hello"), "Hello");
  ASSERT_EQ(pp::fmt("Hello ", 2), "Hello 2");
  ASSERT_EQ(pp::fmt("Hello ", 2 + 2), "Hello 4");
}

TEST(FORMAT, simple_number) { ASSERT_EQ(pp::fmt(2), "2"); }
TEST(FORMAT, concat_lvalue) { ASSERT_EQ(pp::fmt("Hello ", 2), "Hello 2"); }
TEST(FORMAT, concat_rvalue) { ASSERT_EQ(pp::fmt("Hello ", 2 + 2), "Hello 4"); }
