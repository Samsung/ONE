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
struct IsPadF : public coco::Op::Visitor<bool>
{
  bool visit(const coco::PadF *) override { return true; }
};

class PadFTest : public ::testing::Test
{
public:
  PadFTest()
  {
    // DO NOTHING
  }

protected:
  coco::PadF *allocate(void)
  {
    auto op = new coco::PadF;
    _allocated.emplace_back(op);
    return op;
  }

private:
  std::vector<std::unique_ptr<coco::PadF>> _allocated;
};
} // namespace

TEST_F(PadFTest, initialization)
{
  auto op = allocate();

  // uses() should be empty on construction
  ASSERT_EQ(op->uses().size(), 0);
  // parent() should be nullptr on construction
  ASSERT_EQ(op->parent(), nullptr);

  // arg() should be nullptr on construction
  ASSERT_EQ(op->arg(), nullptr);

  // pad() should be a valid
  ASSERT_NE(op->pad(), nullptr);
}

TEST_F(PadFTest, asPadF)
{
  auto op = allocate();

  coco::Op *mutable_base = op;
  const coco::Op *immutable_base = op;

  ASSERT_EQ(mutable_base->asPadF(), op);
  ASSERT_EQ(mutable_base->asPadF(), immutable_base->asPadF());
}

TEST_F(PadFTest, accept)
{
  // Test 'PadF' class
  auto op = allocate();

  coco::PadF *mutable_ptr = op;
  const coco::PadF *immutable_ptr = op;

  ASSERT_TRUE(mutable_ptr->accept(IsPadF{}));
  ASSERT_TRUE(immutable_ptr->accept(IsPadF{}));
}
