/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Optimizer.h"

#include <loco.h>

#include <gtest/gtest.h>

// Optimizer SHOULD NOT crash even though a given graph is empty
TEST(Optimizer, empty_graph)
{
  moco::tf::Optimizer o;

  loco::Graph g;

  o.optimize(&g);

  SUCCEED();
}

TEST(Optimizer, simple_forward_graph)
{
  moco::tf::Optimizer o;

  /**
   * Create a simple graph that forwards a constant as output
   */
  loco::Graph g;
  {
    auto constgen = g.nodes()->create<loco::ConstGen>();
    constgen->shape({2, 3});

    auto forward = g.nodes()->create<loco::Forward>();
    forward->input(constgen);

    auto pull = g.nodes()->create<loco::Push>();
    pull->from(forward);
  }

  o.optimize(&g);

  SUCCEED();
}

TEST(Optimizer, simple_forward_graph_with_one_valid_output)
{
  moco::tf::Optimizer o;

  /**
   * Create a simple graph that forwards a constant as graph-level output
   */
  loco::Graph g;
  {
    auto output = g.outputs()->create();

    auto constgen = g.nodes()->create<loco::ConstGen>();
    constgen->shape({2, 3});
    constgen->dtype(loco::DataType::FLOAT32);
    constgen->size<loco::DataType::FLOAT32>(6);

    auto forward = g.nodes()->create<loco::Forward>();
    forward->input(constgen);

    auto pull = g.nodes()->create<loco::Push>();
    pull->from(forward);

    loco::link(output, pull);
  }

  o.optimize(&g);

  SUCCEED();
}
