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

#include <gtest/gtest.h>

namespace logo
{

void create_empty_test_net(loco::Graph *graph)
{
  assert(graph);

  auto const_node = graph->nodes()->create<loco::ConstGen>();
  {
    const_node->dtype(loco::DataType::FLOAT32);
    const_node->rank(1);
    const_node->dim(0) = 1;
    const_node->size<loco::DataType::FLOAT32>(1);
    const_node->at<loco::DataType::FLOAT32>(0) = 1.0f;
  }

  auto push_node = graph->nodes()->create<loco::Push>();
  {
    push_node->from(const_node);
  }

  auto graph_output = graph->outputs()->create();
  {
    graph_output->name("output");
    graph_output->dtype(loco::DataType::FLOAT32);
    loco::link(graph_output, push_node);
  }
}

} // namespace logo
