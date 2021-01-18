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

#include "TestGraph.h"
#include "CircleTypeInferenceHelper.h"
#include <luci/Service/CircleTypeInferenceRule.h>

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleDialect.h>

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/TypeInference.h>

#include <gtest/gtest.h>

#include <memory>

TEST(CircleTypeInferenceRuleTest, minimal_with_CircleRelu)
{
  // Create a simple network
  luci::test::TestGraph graph;
  auto relu_node = graph.append<luci::CircleRelu>(graph.input_node);
  graph.complete(relu_node);

  // set dtype for I/O nodes
  graph.input_node->dtype(loco::DataType::S32);
  graph.output_node->dtype(loco::DataType::S32);

  luci::test::graph_input_dtype(graph.input_node);
  luci::test::graph_output_dtype(graph.output_node);

  // pre-check
  ASSERT_FALSE(loco::dtype_known(relu_node));

  // type inference
  luci::CircleTypeInferenceRule circle_rule;
  loco::CanonicalTypeInferenceRule canon_rule;
  loco::MultiDialectTypeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canon_rule);
  rules.bind(luci::CircleDialect::get(), &circle_rule);

  loco::apply(&rules).to(graph.g.get());

  // Verify
  ASSERT_TRUE(loco::dtype_known(relu_node));
  auto type = luci::dtype_get(relu_node);
  ASSERT_EQ(loco::DataType::S32, type);
}
