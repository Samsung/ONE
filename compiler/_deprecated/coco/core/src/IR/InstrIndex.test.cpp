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

#include "coco/IR/InstrIndex.h"

#include <gtest/gtest.h>

namespace
{

class InstrIndexTest : public ::testing::Test
{
};

} // namespace

TEST_F(InstrIndexTest, default_constructor)
{
  coco::InstrIndex ins_ind;

  ASSERT_FALSE(ins_ind.valid());
}

TEST_F(InstrIndexTest, explicit_constructor)
{
  coco::InstrIndex ins_ind{3};

  ASSERT_TRUE(ins_ind.valid());
  ASSERT_EQ(ins_ind.value(), 3);
}

TEST_F(InstrIndexTest, operator_lt)
{
  // Valid index is always less than undefined one.
  ASSERT_TRUE(coco::InstrIndex(3) < coco::InstrIndex());
  ASSERT_TRUE(coco::InstrIndex(3) < coco::InstrIndex(4));
}
