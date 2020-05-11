/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodes.h"

#include <loco.h>
#include <logo/DeadNodeQueryService.h>

#include <gtest/gtest.h>

TEST(CircleDialectTest, get_P)
{
  auto d = luci::CircleDialect::get();

  // get() SHOULD return a valid(non-null) pointer
  ASSERT_NE(d, nullptr);
  // The return value SHOULD be stable across multiple invocations
  ASSERT_EQ(luci::CircleDialect::get(), d);
}

TEST(CircleDialectTest, check_if_dead_node_service)
{
  /**
   * [CircleInput1] [CircleInput2]       [CircleInput3]
   *        \           /               (dangling input)
   *         \         /
   *         [CircleAdd]         [CircleBatchMatMul]
   *              |                (dangling node)
   *              |
   *        [CircleOutput1]      [CircleOutput2]
   *                            (dangling output)
   */
  auto g = loco::make_graph();

  auto graph_input1 = g->inputs()->create();
  auto circle_input1 = g->nodes()->create<luci::CircleInput>();
  circle_input1->index(graph_input1->index());

  auto graph_input2 = g->inputs()->create();
  auto circle_input2 = g->nodes()->create<luci::CircleInput>();
  circle_input2->index(graph_input2->index());

  // dangling output
  auto graph_input3 = g->inputs()->create();
  auto dangling_input = g->nodes()->create<luci::CircleInput>();
  dangling_input->index(graph_input3->index());

  auto active_node = g->nodes()->create<luci::CircleAdd>();
  active_node->x(circle_input1);
  active_node->y(circle_input2);

  auto dangling_node = g->nodes()->create<luci::CircleBatchMatMul>();

  auto graph_output1 = g->outputs()->create();
  auto circle_output1 = g->nodes()->create<luci::CircleOutput>();
  circle_output1->index(graph_output1->index());
  circle_output1->from(active_node);

  // dangling output
  auto graph_output2 = g->outputs()->create();
  auto circle_output2 = g->nodes()->create<luci::CircleOutput>();
  circle_output2->index(graph_output2->index());

  auto service = active_node->dialect()->service<logo::DeadNodeQueryService>();

  ASSERT_TRUE(service->isDeadNode(dangling_node));
  ASSERT_FALSE(service->isDeadNode(dangling_input));
  ASSERT_FALSE(service->isDeadNode(active_node));
  ASSERT_FALSE(service->isDeadNode(circle_input1));
  ASSERT_FALSE(service->isDeadNode(circle_input2));
  ASSERT_FALSE(service->isDeadNode(circle_output1));
  ASSERT_FALSE(service->isDeadNode(circle_output2));
}
