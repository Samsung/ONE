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

#include <locoex/Service/COpTypeInference.h>
#include <locoex/COpCall.h>
#include <locoex/COpDialect.h>

#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/TypeInference.h>

#include <gtest/gtest.h>

TEST(TypeInferenceRuleTest, COpTypeInference)
{
  // Create a simple Relu6 network
  auto g = loco::make_graph();

  auto pull_node = g->nodes()->create<loco::Pull>();
  pull_node->dtype(loco::DataType::FLOAT32);

  auto call_node = g->nodes()->create<locoex::COpCall>(1);
  call_node->input(0, pull_node);
  call_node->dtype(loco::DataType::FLOAT32);

  auto push_node = g->nodes()->create<loco::Push>();
  push_node->from(call_node);

  auto graph_input = g->inputs()->create();

  graph_input->name("input");
  loco::link(graph_input, pull_node);

  auto graph_output = g->outputs()->create();

  graph_output->name("output");
  loco::link(graph_output, push_node);

  // Run Type Inference
  locoex::COpTypeInferenceRule cop_rule;
  loco::CanonicalTypeInferenceRule canon_rule;
  loco::MultiDialectTypeInferenceRule rules;

  rules.bind(locoex::COpDialect::get(), &cop_rule).bind(loco::CanonicalDialect::get(), &canon_rule);

  loco::apply(&rules).to(g.get());

  // Verify!
  ASSERT_TRUE(loco::dtype_known(call_node));
  ASSERT_EQ(loco::dtype_get(call_node), loco::DataType::FLOAT32);
}
