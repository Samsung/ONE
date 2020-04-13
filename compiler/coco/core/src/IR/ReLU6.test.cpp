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
struct IsReLU6 : public coco::Op::Visitor<bool>
{
  bool visit(const coco::ReLU6 *) override { return true; }
};

class ReLU6Test : public ::testing::Test
{
public:
  ReLU6Test()
  {
    // DO NOTHING
  }

protected:
  coco::ReLU6 *allocate(void)
  {
    auto op = new coco::ReLU6;
    _allocated.emplace_back(op);
    return op;
  }

private:
  std::vector<std::unique_ptr<coco::ReLU6>> _allocated;
};
} // namespace

TEST_F(ReLU6Test, initialization)
{
  auto op = allocate();

  // uses() should be empty on construction
  ASSERT_EQ(op->uses().size(), 0);
  // parent() should be nullptr on construction
  ASSERT_EQ(op->parent(), nullptr);

  ASSERT_EQ(op->arg(), nullptr);
}

TEST_F(ReLU6Test, asReLU6)
{
  auto op = allocate();

  coco::Op *mutable_base = op;
  const coco::Op *immutable_base = op;

  ASSERT_EQ(mutable_base->asReLU6(), op);
  ASSERT_EQ(mutable_base->asReLU6(), immutable_base->asReLU6());
}

TEST_F(ReLU6Test, accept)
{
  // Test 'ReLU6' class
  auto op = allocate();

  coco::ReLU6 *mutable_ptr = op;
  const coco::ReLU6 *immutable_ptr = op;

  ASSERT_TRUE(mutable_ptr->accept(IsReLU6{}));
  ASSERT_TRUE(immutable_ptr->accept(IsReLU6{}));
}
