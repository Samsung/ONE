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

#include "FuseReluPass.h"

#include "Dialect/IR/TFLNodes.h"
#include "TestGraph.h"

#include <loco.h>
#include <logo/RemoveDeadNodePass.h>

#include <gtest/gtest.h>

#include <type_traits> // for std::is_same

namespace
{

void init(loco::Pull *pull)
{
  pull->dtype(loco::DataType::FLOAT32);
  pull->shape({2, 3, 3, 2});
}

/// @brief Initializes TFLConv2D and related filter and bias
void init(locoex::TFLConv2D *conv2d, locoex::TFLConst *filter, locoex::TFLConst *bias)
{
  // set conv2d
  {
    conv2d->fusedActivationFunction(locoex::FusedActFunc::NONE);
    conv2d->padding(locoex::Padding::VALID);
  }

  // set filter
  {
    filter->dtype(loco::DataType::FLOAT32);
    filter->shape({2, 3, 3, 2});
    filter->size<loco::DataType::FLOAT32>(2 * 3 * 3 * 2);

    for (uint32_t x = 0; x < 2 * 3 * 3 * 2; x++)
      filter->at<loco::DataType::FLOAT32>(x) = 0.0;
  }

  // set bias
  {
    bias->dtype(loco::DataType::FLOAT32);
    bias->shape({2});
    bias->size<loco::DataType::FLOAT32>(2);

    for (uint32_t x = 0; x < 2; x++)
      bias->at<loco::DataType::FLOAT32>(x) = 0.0;
  }
}

} // namespace

/// Test code called by TEST(..)
/// This tests whether Conv2D - FusedTFLType is fused.
template <class FusedTFLType, locoex::FusedActFunc FusedActFunc> void test()
{
  static_assert((std::is_same<FusedTFLType, locoex::TFLRelu>::value &&
                 FusedActFunc == locoex::FusedActFunc::RELU) ||
                  (std::is_same<FusedTFLType, locoex::TFLRelu6>::value &&
                   FusedActFunc == locoex::FusedActFunc::RELU6),
                "wrong template type");

  exo::test::TestGraph g;
  {
    auto filter = g.append<locoex::TFLConst>();
    auto bias = g.append<locoex::TFLConst>();
    auto conv2d = g.append<locoex::TFLConv2D>(g.pull, filter, bias);

    auto fusable_node = g.append<FusedTFLType>(conv2d);

    g.complete(fusable_node);

    init(g.pull);
    init(conv2d, filter, bias);
  }

  // let's run fusion
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::FuseReluPass>();
    test_phase.add_pass<logo::RemoveDeadNodePass>(); // to remove TFLRelu
    test_phase.run(g.graph());
  }

  auto a_conv2d = exo::test::find_first_node_bytype<locoex::TFLConv2D>(g.graph());
  ASSERT_TRUE(a_conv2d != nullptr);
  ASSERT_TRUE(a_conv2d->fusedActivationFunction() == FusedActFunc);

  auto removed_fusable_node = exo::test::find_first_node_bytype<FusedTFLType>(g.graph());
  ASSERT_TRUE(removed_fusable_node == nullptr);
}

// A case with Conv2D-Relu
TEST(FuseReluTest, Conv2D_Relu_basic) { test<locoex::TFLRelu, locoex::FusedActFunc::RELU>(); }

// A case with Conv2D-Relu6
TEST(FuseReluTest, Conv2D_Relu6_basic) { test<locoex::TFLRelu6, locoex::FusedActFunc::RELU6>(); }
