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

#include "ConstGenConverter.h"
#include "ReluConverter.h"

#include "Dialect/IR/TFLNodes.h"
#include "TestGraph.h"
#include "TestHelper.h"

#include <loco.h>

#include <gtest/gtest.h>

TEST(TFLConstGenConverterTest, ConstGen_Relu)
{
  exo::test::ExampleGraph<exo::test::ExampleGraphType::ConstGen_ReLU> g;

  // set constgen
  {
    g.constgen->dtype(loco::DataType::FLOAT32);
    g.constgen->shape({2, 1});
    g.constgen->size<loco::DataType::FLOAT32>(2);

    g.constgen->at<loco::DataType::FLOAT32>(0) = 0.5;
    g.constgen->at<loco::DataType::FLOAT32>(1) = -0.5;
  }

  // let's convert
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::ConstGenConverter>();
    test_phase.add_pass<exo::ReluConverter>();

    test_phase.run(g.graph());
  }

  auto tfl_const = exo::test::find_first_node_bytype<locoex::TFLConst>(g.graph());
  auto tfl_relu = exo::test::find_first_node_bytype<locoex::TFLRelu>(g.graph());

  ASSERT_TRUE(tfl_const != nullptr and tfl_relu != nullptr);
  ASSERT_TRUE(tfl_relu->features() == tfl_const);

  ASSERT_TRUE(tfl_const->rank() == g.constgen->rank());
  ASSERT_TRUE(tfl_const->dim(0) == g.constgen->dim(0));
  ASSERT_TRUE(tfl_const->dim(1) == g.constgen->dim(1));
  ASSERT_TRUE(tfl_const->at<loco::DataType::FLOAT32>(0) ==
              g.constgen->at<loco::DataType::FLOAT32>(0));
  ASSERT_TRUE(tfl_const->at<loco::DataType::FLOAT32>(1) ==
              g.constgen->at<loco::DataType::FLOAT32>(1));
}
