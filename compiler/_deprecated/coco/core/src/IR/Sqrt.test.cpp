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
struct IsSqrt : public coco::Op::Visitor<bool>
{
  bool visit(const coco::Sqrt *) override { return true; }
};

class SqrtTest : public ::testing::Test
{
public:
  SqrtTest()
  {
    // DO NOTHING
  }

protected:
  coco::Sqrt *allocate(void)
  {
    auto op = new coco::Sqrt;
    _allocated.emplace_back(op);
    return op;
  }

private:
  std::vector<std::unique_ptr<coco::Sqrt>> _allocated;
};
} // namespace

TEST_F(SqrtTest, initialization)
{
  auto op = allocate();

  // uses() should be empty on construction
  ASSERT_EQ(op->uses().size(), 0);
  // parent() should be nullptr on construction
  ASSERT_EQ(op->parent(), nullptr);

  ASSERT_EQ(op->arg(), nullptr);
}

TEST_F(SqrtTest, asSqrt)
{
  auto op = allocate();

  coco::Op *mutable_base = op;
  const coco::Op *immutable_base = op;

  ASSERT_EQ(mutable_base->asSqrt(), op);
  ASSERT_EQ(mutable_base->asSqrt(), immutable_base->asSqrt());
}

TEST_F(SqrtTest, accept)
{
  // Test 'Sqrt' class
  auto op = allocate();

  coco::Sqrt *mutable_ptr = op;
  const coco::Sqrt *immutable_ptr = op;

  ASSERT_TRUE(mutable_ptr->accept(IsSqrt{}));
  ASSERT_TRUE(immutable_ptr->accept(IsSqrt{}));
}
