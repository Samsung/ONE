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

#include "locoex/Service/COpShapeInferenceRule.h"
#include "locoex/COpCall.h"
#include <loco/Service/ShapeInference.h>

#include <gtest/gtest.h>

TEST(COpShapeInferenceRuleTest, minimal)
{
  // Create a simple network
  auto g = loco::make_graph();

  auto call_node = g->nodes()->create<locoex::COpCall>(0);
  call_node->shape({1, 3});

  auto push_node = g->nodes()->create<loco::Push>();
  push_node->from(call_node);

  auto graph_output = g->outputs()->create();
  graph_output->name("output");
  loco::link(graph_output, push_node);

  // pre-check
  ASSERT_FALSE(loco::shape_known(call_node));

  // Run Shape Inference
  locoex::COpShapeInferenceRule rule;

  loco::apply(&rule).to(g.get());

  // Verify!
  ASSERT_TRUE(loco::shape_known(call_node));
  ASSERT_EQ(loco::shape_get(call_node).domain(), loco::Domain::Tensor);

  auto shape = loco::shape_get(call_node).as<loco::TensorShape>();
  ASSERT_EQ(shape.rank(), 2);
  ASSERT_EQ(shape.dim(0), 1);
  ASSERT_EQ(shape.dim(1), 3);
}
