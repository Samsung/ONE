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
struct IsSub : public coco::Op::Visitor<bool>
{
  bool visit(const coco::Sub *) override { return true; }
};

class SubTest : public ::testing::Test
{
public:
  SubTest()
  {
    // DO NOTHING
  }

protected:
  coco::Sub *allocate(void)
  {
    auto op = new coco::Sub;
    _allocated.emplace_back(op);
    return op;
  }

private:
  std::vector<std::unique_ptr<coco::Sub>> _allocated;
};
} // namespace

TEST_F(SubTest, initialization)
{
  auto op = allocate();

  // arguments should be empty on construction
  ASSERT_EQ(op->left(), nullptr);
  ASSERT_EQ(op->right(), nullptr);

  // uses() should be empty on construction
  ASSERT_EQ(op->uses().size(), 0);
  // parent() should be nullptr on construction
  ASSERT_EQ(op->parent(), nullptr);
}

TEST_F(SubTest, asSub)
{
  auto op = allocate();

  coco::Op *mutable_base = op;
  const coco::Op *immutable_base = op;

  ASSERT_EQ(mutable_base->asSub(), op);
  ASSERT_EQ(mutable_base->asSub(), immutable_base->asSub());
}

TEST_F(SubTest, accept)
{
  // Test 'Sub' class
  auto op = allocate();

  coco::Sub *mutable_ptr = op;
  const coco::Sub *immutable_ptr = op;

  ASSERT_TRUE(mutable_ptr->accept(IsSub{}));
  ASSERT_TRUE(immutable_ptr->accept(IsSub{}));
}
