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

#include "nest/VarID.h"

#include <gtest/gtest.h>

TEST(VAR_ID, ctor)
{
  nest::VarID id{0};

  ASSERT_EQ(0, id.value());
}

TEST(VAR_ID, operator_eq)
{
  ASSERT_TRUE(nest::VarID(0) == nest::VarID(0));
  ASSERT_FALSE(nest::VarID(0) == nest::VarID(1));
}

TEST(VAR_ID, operator_lt)
{
  ASSERT_TRUE(nest::VarID(0) < nest::VarID(1));
  ASSERT_FALSE(nest::VarID(1) < nest::VarID(0));
}
