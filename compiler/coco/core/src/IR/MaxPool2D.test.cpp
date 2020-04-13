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
struct IsMaxPool2D : public coco::Op::Visitor<bool>
{
  bool visit(const coco::MaxPool2D *) override { return true; }
};

class MaxPool2DTest : public ::testing::Test
{
public:
  MaxPool2DTest()
  {
    // DO NOTHING
  }

protected:
  coco::MaxPool2D *allocate(void)
  {
    auto op = new coco::MaxPool2D;
    _allocated.emplace_back(op);
    return op;
  }

private:
  std::vector<std::unique_ptr<coco::MaxPool2D>> _allocated;
};
} // namespace

TEST_F(MaxPool2DTest, initialization)
{
  auto op = allocate();

  coco::MaxPool2D *mutable_ptr = op;
  const coco::MaxPool2D *immutable_ptr = op;

  // uses() should be empty on construction
  ASSERT_EQ(op->uses().size(), 0);
  // parent() should be nullptr on construction
  ASSERT_EQ(op->parent(), nullptr);

  // arg() should be nullptr on construction
  ASSERT_EQ(immutable_ptr->arg(), nullptr);

  // window() SHOULD return a valid pointer
  ASSERT_NE(mutable_ptr->window(), nullptr);
  ASSERT_EQ(mutable_ptr->window(), immutable_ptr->window());

  // stride() SHOULD return a valid pointer
  ASSERT_NE(mutable_ptr->stride(), nullptr);
  ASSERT_EQ(mutable_ptr->stride(), immutable_ptr->stride());

  // pad() SHOULD return a valid pointer
  ASSERT_NE(mutable_ptr->pad(), nullptr);
  ASSERT_EQ(mutable_ptr->pad(), immutable_ptr->pad());
}

TEST_F(MaxPool2DTest, asMaxPool2D)
{
  auto op = allocate();

  coco::Op *mutable_base = op;
  const coco::Op *immutable_base = op;

  ASSERT_EQ(mutable_base->asMaxPool2D(), op);
  ASSERT_EQ(mutable_base->asMaxPool2D(), immutable_base->asMaxPool2D());
}

TEST_F(MaxPool2DTest, accept)
{
  // Test 'MaxPool2D' class
  auto op = allocate();

  coco::MaxPool2D *mutable_ptr = op;
  const coco::MaxPool2D *immutable_ptr = op;

  ASSERT_TRUE(mutable_ptr->accept(IsMaxPool2D{}));
  ASSERT_TRUE(immutable_ptr->accept(IsMaxPool2D{}));
}
