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
struct IsReLU : public coco::Op::Visitor<bool>
{
  bool visit(const coco::ReLU *) override { return true; }
};

class ReLUTest : public ::testing::Test
{
public:
  ReLUTest()
  {
    // DO NOTHING
  }

protected:
  coco::ReLU *allocate(void)
  {
    auto op = new coco::ReLU;
    _allocated.emplace_back(op);
    return op;
  }

private:
  std::vector<std::unique_ptr<coco::ReLU>> _allocated;
};
} // namespace

TEST_F(ReLUTest, initialization)
{
  auto op = allocate();

  // uses() should be empty on construction
  ASSERT_EQ(op->uses().size(), 0);
  // parent() should be nullptr on construction
  ASSERT_EQ(op->parent(), nullptr);

  ASSERT_EQ(op->arg(), nullptr);
}

TEST_F(ReLUTest, asReLU)
{
  auto op = allocate();

  coco::Op *mutable_base = op;
  const coco::Op *immutable_base = op;

  ASSERT_EQ(mutable_base->asReLU(), op);
  ASSERT_EQ(mutable_base->asReLU(), immutable_base->asReLU());
}

TEST_F(ReLUTest, accept)
{
  // Test 'ReLU' class
  auto op = allocate();

  coco::ReLU *mutable_ptr = op;
  const coco::ReLU *immutable_ptr = op;

  ASSERT_TRUE(mutable_ptr->accept(IsReLU{}));
  ASSERT_TRUE(immutable_ptr->accept(IsReLU{}));
}
