/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "loco/IR/Use.h"

#include "MockupNode.h"

#include <gtest/gtest.h>

TEST(UseTest, constructor)
{
  MockupNode user;
  loco::Use use{&user};

  ASSERT_EQ(use.user(), &user);
  ASSERT_EQ(use.node(), nullptr);
}

TEST(UseTest, link_node)
{
  MockupNode def;
  MockupNode user;
  loco::Use use{&user};

  use.node(&def);

  ASSERT_EQ(use.user(), &user);
  ASSERT_EQ(use.node(), &def);
}
