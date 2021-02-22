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

#include "Dialect/IR/TFLNodes.h"
#include "Dialect/IR/TFLDialect.h"
#include "Dialect/Service/TFLTypeInferenceRule.h"

#include "TestGraph.h"

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/TypeInference.h>

#include <gtest/gtest.h>

TEST(TFLTypeInferenceRuleTest, minimal_with_TFLRelu)
{
  // Create a simple network
  exo::test::TestGraph graph;
  auto tfl_node = graph.append<locoex::TFLRelu>(graph.pull);
  graph.complete(tfl_node);

  graph.pull->dtype(loco::DataType::S32);

  // pre-check
  ASSERT_FALSE(loco::dtype_known(tfl_node));

  // type inference
  locoex::TFLTypeInferenceRule tfl_rule;
  loco::CanonicalTypeInferenceRule canon_rule;
  loco::MultiDialectTypeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canon_rule);
  rules.bind(locoex::TFLDialect::get(), &tfl_rule);

  loco::apply(&rules).to(graph.g.get());

  // Verify
  ASSERT_TRUE(loco::dtype_known(tfl_node));
  auto type = loco::dtype_get(tfl_node);
  ASSERT_EQ(loco::DataType::S32, type);
}
