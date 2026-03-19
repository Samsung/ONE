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

#include "TestHelper.h"

namespace moco
{
namespace test
{

void setup_output_node(loco::Graph *graph, loco::Node *last_node)
{
  // add push as output
  auto push_node = graph->nodes()->create<loco::Push>();
  push_node->from(last_node);

  // set the graph output name and node object
  auto graph_output = graph->outputs()->create();
  graph_output->name("output");
  graph_output->dtype(loco::DataType::FLOAT32);
  loco::link(graph_output, push_node);
}

} // namespace test
} // namespace moco
