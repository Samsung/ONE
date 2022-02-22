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

#include "luci/Plan/CircleNodeExecutionPlan.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

TEST(CircleNodeExecutionPlan, basic_fields)
{
  luci::CircleNodeExecutionPlan plan(123, {4, 5, 6, 7});

  ASSERT_EQ(plan.order_in_plan(), 123);
  ASSERT_THAT(plan.offsets(), testing::ElementsAre(4, 5, 6, 7));

  plan.order_in_plan(321);
  plan.offsets({1, 2, 3, 4});

  ASSERT_EQ(plan.order_in_plan(), 321);
  ASSERT_THAT(plan.offsets(), testing::ElementsAre(1, 2, 3, 4));
}

TEST(CircleNodeExecutionPlan, add_extract_plan)
{
  auto g = loco::make_graph();
  auto add = g->nodes()->create<luci::CircleAdd>();

  ASSERT_FALSE(luci::has_execution_plan(add));

  luci::CircleNodeExecutionPlan plan(123, {4, 5, 6, 7});
  luci::add_execution_plan(add, plan);

  ASSERT_TRUE(luci::has_execution_plan(add));

  auto extracted_plan = luci::get_execution_plan(add);

  ASSERT_EQ(extracted_plan.order_in_plan(), 123);
  ASSERT_THAT(extracted_plan.offsets(), testing::ElementsAre(4, 5, 6, 7));
}

TEST(CircleNodeExecutionPlan, extract_plan_NEG)
{
  auto g = loco::make_graph();
  auto add = g->nodes()->create<luci::CircleAdd>();

  ASSERT_FALSE(luci::has_execution_plan(add));

  ASSERT_ANY_THROW(luci::get_execution_plan(add));
}

TEST(CircleNodeExecutionPlan, double_set_plan_NEG)
{
  auto g = loco::make_graph();
  auto add = g->nodes()->create<luci::CircleAdd>();

  ASSERT_FALSE(luci::has_execution_plan(add));

  luci::CircleNodeExecutionPlan plan1(123, {4, 5, 6, 7});
  luci::add_execution_plan(add, plan1);
  ASSERT_TRUE(luci::has_execution_plan(add));

  luci::CircleNodeExecutionPlan plan2(321, {1, 2, 3, 4});
  luci::add_execution_plan(add, plan2);
  ASSERT_TRUE(luci::has_execution_plan(add));

  auto extracted_plan = luci::get_execution_plan(add);
  ASSERT_EQ(extracted_plan.order_in_plan(), 321);
  ASSERT_THAT(extracted_plan.offsets(), testing::ElementsAre(1, 2, 3, 4));
}
