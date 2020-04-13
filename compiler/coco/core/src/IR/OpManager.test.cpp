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

#include "coco/IR/OpManager.h"

#include <gtest/gtest.h>

namespace
{

class OpManagerTest : public ::testing::Test
{
protected:
  coco::OpManager mgr;
};

} // namespace

TEST(IR_OP_MANAGER, create_Conv2D)
{
  coco::OpManager mgr;

  auto obj = mgr.create<coco::Conv2D>();

  ASSERT_NE(obj, nullptr);
}

TEST(IR_OP_MANAGER, create_AvgPool2D)
{
  coco::OpManager mgr;

  auto obj = mgr.create<coco::AvgPool2D>();

  ASSERT_NE(obj, nullptr);
}

TEST_F(OpManagerTest, ReLU)
{
  auto obj = mgr.create<coco::ReLU>();

  ASSERT_NE(obj, nullptr);
}

TEST_F(OpManagerTest, ReLU6)
{
  auto obj = mgr.create<coco::ReLU6>();

  ASSERT_NE(obj, nullptr);
}

TEST_F(OpManagerTest, Sqrt)
{
  auto obj = mgr.create<coco::Sqrt>();

  ASSERT_NE(obj, nullptr);
}

TEST_F(OpManagerTest, Sub)
{
  auto obj = mgr.create<coco::Sub>();

  ASSERT_NE(obj, nullptr);
}

TEST_F(OpManagerTest, Div)
{
  auto obj = mgr.create<coco::Div>();

  ASSERT_NE(obj, nullptr);
}

TEST_F(OpManagerTest, PadF)
{
  auto op = mgr.create<coco::PadF>();
  ASSERT_NE(op, nullptr);
  mgr.destroy(op);
}

TEST_F(OpManagerTest, destroy)
{
  auto op = mgr.create<coco::Conv2D>();
  mgr.destroy(op);
  ASSERT_EQ(mgr.size(), 0);
}

TEST_F(OpManagerTest, destroy_all)
{
  // Create a Op tree
  auto load_op = mgr.create<coco::Load>();
  auto conv_op = mgr.create<coco::Conv2D>();

  conv_op->arg(load_op);

  mgr.destroy_all(conv_op);

  ASSERT_EQ(mgr.size(), 0);
}

TEST_F(OpManagerTest, destroy_all_partial_tree)
{
  // Create a (partial) Op tree
  auto conv_op = mgr.create<coco::Conv2D>();

  mgr.destroy_all(conv_op);

  ASSERT_EQ(mgr.size(), 0);
}
