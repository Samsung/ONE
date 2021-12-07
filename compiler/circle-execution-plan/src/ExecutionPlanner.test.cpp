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

#include <loco.h>
#include "ExecutionPlanner.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

TEST(ExecutionPlannerTest, op_with_multiple_outputs)
{
  auto g = loco::make_graph();

  // Create nodes
  auto input = g->nodes()->create<luci::CircleInput>();
  auto split = g->nodes()->create<luci::CircleSplit>();
  auto split_out1 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_out2 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_dim = g->nodes()->create<luci::CircleConst>();

  // Build up initial graph
  auto graph_input1 = g->inputs()->create();
  graph_input1->shape({1, 6});

  input->index(graph_input1->index());
  input->shape({1, 6});
  input->dtype(loco::DataType::S32);
  input->shape_status(luci::ShapeStatus::VALID);
  input->name("input");

  split_dim->dtype(loco::DataType::S32);
  split_dim->size<loco::DataType::S32>(1);
  split_dim->shape({1});
  split_dim->at<loco::DataType::S32>(0) = 1;
  split_dim->shape_status(luci::ShapeStatus::VALID);
  split_dim->name("split_dim_const");

  split->split_dim(split_dim);
  split->dtype(input->dtype());
  split->input(input);
  split->num_split(2);
  split->name("split");

  split_out1->input(split);
  split_out1->index(0);
  split_out1->shape({1, 3});
  split_out1->dtype(split->dtype());
  split_out1->name("split_out1");

  split_out2->input(split);
  split_out2->index(1);
  split_out2->shape({1, 3});
  split_out2->dtype(split->dtype());
  split_out2->name("split_out2");

  auto add = g->nodes()->create<luci::CircleAdd>();
  add->name("add");
  add->x(split_out1);
  add->y(split_out2);
  add->dtype(split_out1->dtype());
  add->shape({1, 3});

  auto output = g->nodes()->create<luci::CircleOutput>();
  output->name("output");
  output->from(add);
  output->dtype(split_out1->dtype());
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
  graph_output->shape({1, 3});

  // Create ExecutionPlanner
  circle_planner::ExecutionPlanner execution_planner(g.get());
  execution_planner.make_execution_plan();

  // Check result
  auto split_out_1_exec_plan = luci::get_execution_plan(split_out1);
  auto split_out_2_exec_plan = luci::get_execution_plan(split_out2);

  ASSERT_TRUE(split_out_1_exec_plan.offsets().front() != split_out_2_exec_plan.offsets().front());

  SUCCEED();
}

TEST(ExecutionPlannerTest, graph_with_two_outputs)
{
  auto g = loco::make_graph();

  // Create nodes
  auto input = g->nodes()->create<luci::CircleInput>();
  auto split = g->nodes()->create<luci::CircleSplit>();
  auto split_out1 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_out2 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_dim = g->nodes()->create<luci::CircleConst>();

  // Build up initial graph
  auto graph_input1 = g->inputs()->create();
  graph_input1->shape({1, 8});

  input->index(graph_input1->index());
  input->shape({1, 8});
  input->dtype(loco::DataType::S32);
  input->shape_status(luci::ShapeStatus::VALID);
  input->name("input");

  split_dim->dtype(loco::DataType::S32);
  split_dim->size<loco::DataType::S32>(1);
  split_dim->shape({1});
  split_dim->at<loco::DataType::S32>(0) = 1;
  split_dim->shape_status(luci::ShapeStatus::VALID);
  split_dim->name("split_dim_const");

  split->split_dim(split_dim);
  split->dtype(input->dtype());
  split->input(input);
  split->num_split(2);
  split->name("split");

  split_out1->input(split);
  split_out1->index(0);
  split_out1->shape({1, 4});
  split_out1->dtype(split->dtype());
  split_out1->name("split_out1");

  split_out2->input(split);
  split_out2->index(1);
  split_out2->shape({1, 4});
  split_out2->dtype(split->dtype());
  split_out2->name("split_out2");

  auto output1 = g->nodes()->create<luci::CircleOutput>();
  output1->name("output1");
  output1->from(split_out1);
  output1->dtype(split_out1->dtype());
  output1->shape({1, 4});
  auto graph_output1 = g->outputs()->create();
  output1->index(graph_output1->index());
  graph_output1->shape({1, 4});

  auto output2 = g->nodes()->create<luci::CircleOutput>();
  output2->name("output2");
  output2->from(split_out2);
  output2->dtype(split_out2->dtype());
  output2->shape({1, 4});
  auto graph_output2 = g->outputs()->create();
  output2->index(graph_output2->index());
  graph_output2->shape({1, 4});

  // Create ExecutionPlanner
  circle_planner::ExecutionPlanner execution_planner(g.get());
  execution_planner.make_execution_plan();

  // Check result
  auto output1_exec_plan = luci::get_execution_plan(output1);
  auto output2_exec_plan = luci::get_execution_plan(output2);

  ASSERT_TRUE(output1_exec_plan.offsets().front() != output2_exec_plan.offsets().front());
  ASSERT_TRUE(output1_exec_plan.offsets().front() + output1->dim(1).value() <=
              output2_exec_plan.offsets().front() + output2->dim(1).value());

  SUCCEED();
}
