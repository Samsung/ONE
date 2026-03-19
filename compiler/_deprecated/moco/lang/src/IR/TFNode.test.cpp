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

#include "moco/IR/Nodes/TFPlaceholder.h"

#include <loco.h>

#include <gtest/gtest.h>

TEST(TFNodeTest_Placeholder, index)
{
  loco::Graph graph;

  auto test_node = graph.nodes()->create<moco::TFPlaceholder>();

  loco::GraphInputIndex index_set{100};
  moco::index(test_node, index_set);

  auto index_get = moco::index(test_node);
  ASSERT_EQ(index_get, index_set);
}

TEST(TFNodeTest_Placeholder, name)
{
  loco::Graph graph;

  auto test_node = graph.nodes()->create<moco::TFPlaceholder>();

  test_node->name("PlaceholderName");
  ASSERT_EQ(test_node->name(), "PlaceholderName");
}
