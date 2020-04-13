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

#include "coco/IR/Instrs.h"
#include "coco/IR/ObjectManager.h"
#include "coco/IR/OpManager.h"

#include <gtest/gtest.h>

namespace
{
class ShuffleTest : public ::testing::Test
{
public:
  virtual ~ShuffleTest() = default;

protected:
  coco::Shuffle *allocate(void)
  {
    auto ins = new coco::Shuffle;
    _allocated.emplace_back(ins);
    return ins;
  }

private:
  std::vector<std::unique_ptr<coco::Instr>> _allocated;
};
} // namespace

TEST_F(ShuffleTest, constructor)
{
  auto ins = allocate();

  ASSERT_EQ(ins->from(), nullptr);
  ASSERT_EQ(ins->into(), nullptr);
}

TEST_F(ShuffleTest, asShuffle)
{
  auto ins = allocate();

  coco::Instr *mutable_ptr = ins;
  const coco::Instr *immutable_ptr = ins;

  ASSERT_NE(mutable_ptr->asShuffle(), nullptr);
  ASSERT_EQ(mutable_ptr->asShuffle(), immutable_ptr->asShuffle());
}

TEST_F(ShuffleTest, size)
{
  auto shuffle = allocate();

  shuffle->insert(coco::ElemID{3}, coco::ElemID{2});
  shuffle->insert(coco::ElemID{3}, coco::ElemID{5});

  ASSERT_EQ(shuffle->size(), 2);
  ASSERT_EQ(shuffle->range().size(), shuffle->size());
}

TEST_F(ShuffleTest, range)
{
  auto shuffle = allocate();

  shuffle->insert(coco::ElemID{3}, coco::ElemID{2});
  shuffle->insert(coco::ElemID{3}, coco::ElemID{5});

  auto range = shuffle->range();

  EXPECT_EQ(range.size(), 2);
  EXPECT_NE(range.count(coco::ElemID{2}), 0);
  EXPECT_NE(range.count(coco::ElemID{5}), 0);
}

TEST_F(ShuffleTest, defined)
{
  auto shuffle = allocate();

  shuffle->insert(coco::ElemID{3}, coco::ElemID{2});

  EXPECT_TRUE(shuffle->defined(coco::ElemID{2}));
  EXPECT_FALSE(shuffle->defined(coco::ElemID{3}));
}
