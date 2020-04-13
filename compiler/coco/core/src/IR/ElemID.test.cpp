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

#include "coco/IR/ElemID.h"

#include <vector>

#include <gtest/gtest.h>

TEST(IR_ELEM_ID, constructor)
{
  coco::ElemID id{128};

  ASSERT_EQ(id.value(), 128);
}

TEST(IR_ELEM_ID, copy)
{
  coco::ElemID src{16};
  coco::ElemID dst{32};

  dst = src;

  ASSERT_EQ(dst.value(), 16);
}

TEST(IR_ELEM_ID, std_vector_compatible)
{
  // ElemID SHOULD be compatible with standard container (including std::vector)
  std::vector<coco::ElemID> vec;

  vec.resize(16);
  vec.clear();
  vec.emplace_back(coco::ElemID{128});

  ASSERT_EQ(vec.at(0).value(), 128);
}

TEST(IR_ELEM_ID, operator_eq)
{
  ASSERT_TRUE(coco::ElemID{16} == coco::ElemID{16});
  ASSERT_FALSE(coco::ElemID{16} == coco::ElemID{17});
}

TEST(IR_ELEM_ID, operator_lt)
{
  ASSERT_FALSE(coco::ElemID{16} < coco::ElemID{16});
  ASSERT_TRUE(coco::ElemID{16} < coco::ElemID{17});
}
