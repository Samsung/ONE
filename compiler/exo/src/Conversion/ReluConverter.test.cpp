/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ReluConverter.h"

#include "GraphBlock.h"
#include "Dialect/IR/TFLNodes.h"

#include "TestHelper.h"
#include "TestGraph.h"

#include <gtest/gtest.h>

TEST(ReluConverterTest, relu_tensor_inout)
{
  exo::test::TestGraph graph;
  {
    auto tanh = graph.append<loco::Tanh>(graph.pull);
    auto relu = graph.append<loco::ReLU>(tanh);
    auto relu6 = graph.append<loco::ReLU6>(relu);
    graph.complete();

    auto pull = graph.pull;
    {
      pull->dtype(loco::DataType::FLOAT32);
      pull->shape({2, 2});
    }
  }

  // let's convert
  exo::test::TypeShapeReadyPhase test_phase;
  {
    test_phase.add_pass<exo::ReluConverter>();
    test_phase.run(graph.g.get());
  }

  loco::Node *node = exo::test::find_first_node_bytype<loco::Tanh>(graph.g.get());
  ASSERT_TRUE(node != nullptr);
  node = exo::test::get_only_succ<locoex::TFLRelu>(node);
  ASSERT_TRUE(node != nullptr);
  node = exo::test::get_only_succ<loco::ReLU6>(node);
  ASSERT_TRUE(node != nullptr);
}

TEST(ReluConverterTest, relu_feature_inout)
{
  // g = Pull - FeatureEncode - Relu - FeatureDecode - Push
  exo::test::TestGraph graph;
  {
    auto enc = exo::make_feature_encode<exo::FeatureLayout::NHWC>(graph.pull);
    auto relu = graph.append<loco::ReLU>(enc);
    auto dec = exo::make_feature_decode<exo::FeatureLayout::NHWC>(relu);
    graph.complete(dec);
  }

  auto pull = graph.pull;
  {
    pull->dtype(loco::DataType::FLOAT32);
    pull->shape({1, 2, 3, 4});
  }

  exo::test::TypeShapeReadyPhase test_phase;
  {
    test_phase.add_pass<exo::ReluConverter>();
    test_phase.run(graph.g.get());
  }

  // now, g = Pull - FeatureEncode - FeatureDecode - TFLRelu - FeatureEncode - FeatureDecode - Push

  // Check
  EXO_TEST_ASSERT_NODE_COUNT({graph.push}, 7);

  // Check [FeatureEncode - FeatureDecode - TFLRelu - FeatureEncode - FeatureDecode] chunk
  loco::Node *node = exo::test::find_first_node_bytype<loco::FeatureEncode>(graph.g.get());
  ASSERT_TRUE(node != nullptr);
  node = exo::test::get_only_succ<loco::FeatureDecode>(node);
  ASSERT_TRUE(node != nullptr);
  node = exo::test::get_only_succ<locoex::TFLRelu>(node);
  ASSERT_TRUE(node != nullptr);
  node = exo::test::get_only_succ<loco::FeatureEncode>(node);
  ASSERT_TRUE(node != nullptr);
  node = exo::test::get_only_succ<loco::FeatureDecode>(node);
  ASSERT_TRUE(node != nullptr);
}
