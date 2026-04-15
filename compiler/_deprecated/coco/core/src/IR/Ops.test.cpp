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

#include "coco/IR/Ops.h"
#include "coco/IR/ObjectManager.h"
#include "coco/IR/OpManager.h"

#include <vector>
#include <memory>

#include <gtest/gtest.h>

using std::make_unique;

/**
 * Section: Add Op
 */
namespace
{

class AddTest : public ::testing::Test
{
public:
  AddTest()
  {
    // DO NOTHING
  }

protected:
  coco::Add *allocate(void)
  {
    auto op = new coco::Add;
    _allocated.emplace_back(op);
    return op;
  }

protected:
  coco::ObjectManager obj_mgr;

private:
  std::vector<std::unique_ptr<coco::Op>> _allocated;
};

} // namespace

TEST_F(AddTest, constructor)
{
  auto op = allocate();

  ASSERT_EQ(op->left(), nullptr);
  ASSERT_EQ(op->right(), nullptr);
}

/**
 * Section: Mul Op
 */
TEST(MulTest, constructor)
{
  auto op = make_unique<coco::Mul>();

  ASSERT_EQ(op->left(), nullptr);
  ASSERT_EQ(op->right(), nullptr);
}

/**
 * Section: Div Op
 */
TEST(DivTest, constructor)
{
  auto op = make_unique<coco::Div>();

  ASSERT_EQ(op->left(), nullptr);
  ASSERT_EQ(op->right(), nullptr);
}

/**
 * Section: Op Helpers
 */
namespace
{

class OpHelperTest : public ::testing::Test
{
public:
  OpHelperTest()
  {
    // DO NOTHING
  }

protected:
  template <typename Op> Op *allocate(void) { return op_mgr.create<Op>(); }

protected:
  coco::ObjectManager obj_mgr;

private:
  coco::OpManager op_mgr;
};

} // namespace

TEST_F(OpHelperTest, root)
{
  auto load = allocate<coco::Load>();

  ASSERT_EQ(root(load), load);

  auto avgpool = allocate<coco::AvgPool2D>();

  avgpool->arg(load);

  ASSERT_EQ(root(load), avgpool);
  ASSERT_EQ(root(avgpool), avgpool);
}
