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

#include "FeatureBiasAddConverter.h"

#include "GraphBlock.h"
#include "Dialect/IR/TFLNodes.h"

#include "TestGraph.h"
#include "TestHelper.h"

#include <loco.h>

#include <gtest/gtest.h>

TEST(FeatureBiasAddConverterTest, basic_test)
{
  exo::test::ExampleGraph<exo::test::ExampleGraphType::FeatureBiasAdd> g;

  { // attrib setting
    // pull
    g.pull->dtype(loco::DataType::FLOAT32);
    g.pull->shape({1, 2, 2, 3});

    // bias value
    g.constgen->dtype(loco::DataType::FLOAT32);
    g.constgen->shape({3});
    g.constgen->size<loco::DataType::FLOAT32>(3);

    g.constgen->at<loco::DataType::FLOAT32>(0) = 0.5;
    g.constgen->at<loco::DataType::FLOAT32>(1) = 1;
    g.constgen->at<loco::DataType::FLOAT32>(2) = 1.5;
  }

  EXO_TEST_ASSERT_NODE_COUNT({g.push}, 7); // sanity check

  // let's convert!!
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::FeatureBiasAddConverter>();

    test_phase.run(g.graph());

    /*
    Expected:

        Pull - FeatureEncoder - FeatureDecode - TFLAdd - FeatureEncode - FeatureDecode - Push
                                                |
          ConstGen - BiasEncode - BiasDecode ---+
    */
  }

  // check surroundings
  auto tfl_add = exo::test::find_first_node_bytype<locoex::TFLAdd>(g.graph());
  {
    ASSERT_TRUE(tfl_add != nullptr);

    // input x and its pred
    {
      auto actual_fea_dec = dynamic_cast<loco::FeatureDecode *>(tfl_add->x());
      ASSERT_TRUE(actual_fea_dec != nullptr);

      auto actual_fea_enc = dynamic_cast<loco::FeatureEncode *>(actual_fea_dec->input());
      ASSERT_TRUE(actual_fea_enc != nullptr);
      ASSERT_TRUE(actual_fea_enc == g.fea_enc);
    }

    // input y and its pred
    {
      auto actual_bias_dec = dynamic_cast<loco::BiasDecode *>(tfl_add->y());
      ASSERT_TRUE(actual_bias_dec != nullptr);

      auto actual_bias_enc = dynamic_cast<loco::BiasEncode *>(actual_bias_dec->input());
      ASSERT_TRUE(actual_bias_enc != nullptr);
      ASSERT_TRUE(actual_bias_enc == g.bias_enc);
    }

    // output check
    {
      auto actual_fea_enc = exo::test::get_only_succ<loco::FeatureEncode>(tfl_add);
      ASSERT_TRUE(actual_fea_enc != nullptr);

      auto actual_fea_dec = exo::test::get_only_succ<loco::FeatureDecode>(actual_fea_enc);
      ASSERT_TRUE(actual_fea_dec != nullptr);
      ASSERT_TRUE(actual_fea_dec == g.fea_dec);
    }
  }
}
