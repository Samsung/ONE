/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Service/Nodes/CircleConst.h"

#include <loco.h>
#include <loco/IR/Graph.h>

#include <gtest/gtest.h>

TEST(CirCleConst, clone)
{
  auto g = loco::make_graph();

  // prepare source CircleConst
  auto circle_const = g->nodes()->create<luci::CircleConst>();

  const auto size = 2;

  circle_const->dtype(loco::DataType::S32);
  circle_const->rank(1);
  circle_const->dim(0).set(size);
  circle_const->shape_status(luci::ShapeStatus::VALID);

  circle_const->size<loco::DataType::S32>(size);
  for (uint32_t i = 0; i < size; i++)
    circle_const->at<loco::DataType::S32>(i) = i;

  // make a clone
  auto const_cloned = luci::clone(circle_const);

  // check attributes
  ASSERT_EQ(loco::DataType::S32, const_cloned->dtype());
  ASSERT_EQ(1, const_cloned->rank());
  ASSERT_EQ(2, const_cloned->dim(0).value());
  ASSERT_EQ(2, const_cloned->size<loco::DataType::S32>());
  ASSERT_EQ(0, const_cloned->at<loco::DataType::S32>(0));
  ASSERT_EQ(1, const_cloned->at<loco::DataType::S32>(1));
  ASSERT_EQ(nullptr, const_cloned->quantparam());
  ASSERT_EQ(nullptr, const_cloned->sparsityparam());
}
