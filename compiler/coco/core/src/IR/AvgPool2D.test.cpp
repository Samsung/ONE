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

#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace
{
struct IsAvgPool2D : public coco::Op::Visitor<bool>
{
  bool visit(const coco::AvgPool2D *) override { return true; }
};

class AvgPool2DTest : public ::testing::Test
{
public:
  AvgPool2DTest()
  {
    // DO NOTHING
  }

protected:
  coco::AvgPool2D *allocate(void)
  {
    auto op = new coco::AvgPool2D;
    _allocated.emplace_back(op);
    return op;
  }

private:
  std::vector<std::unique_ptr<coco::AvgPool2D>> _allocated;
};
} // namespace

TEST_F(AvgPool2DTest, initialization)
{
  auto op = allocate();

  coco::AvgPool2D *mutable_ptr = op;
  const coco::AvgPool2D *immutable_ptr = op;

  // uses() should be empty on construction
  ASSERT_EQ(op->uses().size(), 0);
  // parent() should be nullptr on construction
  ASSERT_EQ(op->parent(), nullptr);

  // arg() should be nullptr on construction
  ASSERT_EQ(immutable_ptr->arg(), nullptr);

  // divisor() SHOULD be unknow on construction
  ASSERT_EQ(immutable_ptr->divisor(), coco::AvgPool2D::Divisor::Unknown);

  // window() SHOULD return a valid pointer
  ASSERT_NE(mutable_ptr->window(), nullptr);
  ASSERT_EQ(mutable_ptr->window(), immutable_ptr->window());

  // pad() SHOULD return a valid pointer
  ASSERT_NE(mutable_ptr->pad(), nullptr);
  ASSERT_EQ(mutable_ptr->pad(), immutable_ptr->pad());

  // stride() SHOULD return a valid pointer
  ASSERT_NE(mutable_ptr->stride(), nullptr);
  ASSERT_EQ(mutable_ptr->stride(), immutable_ptr->stride());
}

TEST_F(AvgPool2DTest, asAvgPool2D)
{
  auto op = allocate();

  coco::Op *mutable_base = op;
  const coco::Op *immutable_base = op;

  ASSERT_EQ(mutable_base->asAvgPool2D(), op);
  ASSERT_EQ(mutable_base->asAvgPool2D(), immutable_base->asAvgPool2D());
}

TEST_F(AvgPool2DTest, accept)
{
  // Test 'AvgPool2D' class
  auto op = allocate();

  coco::AvgPool2D *mutable_ptr = op;
  const coco::AvgPool2D *immutable_ptr = op;

  ASSERT_TRUE(mutable_ptr->accept(IsAvgPool2D{}));
  ASSERT_TRUE(immutable_ptr->accept(IsAvgPool2D{}));
}

TEST_F(AvgPool2DTest, disivor)
{
  auto op = allocate();

  op->divisor(coco::AvgPool2D::Divisor::Static);

  ASSERT_EQ(op->divisor(), coco::AvgPool2D::Divisor::Static);
}
