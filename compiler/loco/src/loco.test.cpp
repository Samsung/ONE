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

#include "loco.h"

#include <gtest/gtest.h>

// This test shows how to create an "identity" network with loco.
//
// What is "identity" network?
// - A network simply passes its input as its output
//
// TODO Create "Ouput" first and then create "Push" later
TEST(LOCO, identity_network)
{
  auto g = loco::make_graph();

  // Create a "pull" node as an input
  auto pull_node = g->nodes()->create<loco::Pull>();

  // Set "data type"
  pull_node->dtype(loco::DataType::FLOAT32);

  // Set "data shape"
  pull_node->rank(2);
  pull_node->dim(0) = 3;
  pull_node->dim(1) = 4;

  // Create a "push" node as an output
  auto push_node = g->nodes()->create<loco::Push>();

  // Set "source"
  push_node->from(pull_node);

  // Create Graph Input & Output
  auto graph_input = g->inputs()->create();

  graph_input->name("input");
  loco::link(graph_input, pull_node);
  graph_input->dtype(loco::DataType::FLOAT32);

  auto graph_output = g->outputs()->create();

  graph_output->name("output");
  loco::link(graph_output, push_node);

  // loco::link SHOULD update "index"
  ASSERT_EQ(pull_node->index(), 0);
  ASSERT_EQ(graph_input->dtype(), loco::DataType::FLOAT32);

  // loco::link SHOULD update "index"
  ASSERT_EQ(push_node->index(), 0);
}

#if 0
"identity_network_V2" test shows how to use loco when loco.core and loco.canonical are decoupled.

NOTE "identity_network" test is left for backward compatiblity check
TODO Remove "identity_network" test once all the clients are migrated.
#endif
TEST(LOCO, identity_network_V2)
{
  auto g = loco::make_graph();

  // Create Graph Input & Output
  auto graph_input = g->inputs()->create();

  graph_input->name("input");
  graph_input->dtype(loco::DataType::FLOAT32);
  // TODO Set Shape

  auto graph_output = g->outputs()->create();

  graph_output->name("output");
  graph_output->dtype(loco::DataType::FLOAT32);
  // TODO Set Shape

  // Create a "pull" node as an input
  auto pull_node = g->nodes()->create<loco::Pull>();

  pull_node->index(0);

  // Create a "push" node as an output
  auto push_node = g->nodes()->create<loco::Push>();

  push_node->index(0);
  push_node->from(pull_node);

  ASSERT_EQ(pull_node->dtype(), loco::DataType::FLOAT32);
  // TODO Check Shape of pull_node
  // TODO Check Shape of push_node

  ASSERT_EQ(loco::pull_node(g.get(), 0), pull_node);
  ASSERT_EQ(loco::push_node(g.get(), 0), push_node);
}
