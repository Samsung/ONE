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

#include "FuseBiasAddPass.h"

#include "Dialect/IR/TFLNodes.h"
#include "TestGraph.h"
#include "TestHelper.h"

#include <loco.h>

#include <gtest/gtest.h>

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

template <class T> void init(T *node, locoex::FusedActFunc f)
{
  static_assert(std::is_same<T, locoex::TFLAdd>::value || std::is_same<T, locoex::TFLSub>::value,
                "wrong template type");

  node->fusedActivationFunction(f);
}

/// @brief Initializes one param of TFLAdd or TFLSub
void init(locoex::TFLConst *addsub_param)
{
  // set addsub_param : y() value of TFLAdd or TFLSub
  addsub_param->dtype(loco::DataType::FLOAT32);
  addsub_param->shape({2});
  addsub_param->size<loco::DataType::FLOAT32>(2);

  for (uint32_t x = 0; x < 2; x++)
    addsub_param->at<loco::DataType::FLOAT32>(x) = (x + 1) * 1.5; // 1.5, 3
}

} // namespace

// A case when
// - TFLConv2D has bias (0, 0)
// - TFLAdd, of which x() or y() == TFLConv2D
// - Another param of TFLAdd is TFLConst, (1.5, 3)
//
// After fusion, bias shold be (1.5, 3)
TEST(FuseBiasAddPassTest, Conv2D_Add_01_basic)
{
  exo::test::TestGraph g;
  auto filter = g.append<locoex::TFLConst>();
  auto bias = g.append<locoex::TFLConst>();
  auto conv2d = g.append<locoex::TFLConv2D>(g.pull, filter, bias);

  auto add_y = g.append<locoex::TFLConst>();
  auto add = g.append<locoex::TFLAdd>(conv2d, add_y);

  g.complete(add);

  init(g.pull);
  init(conv2d, filter, bias);
  init(add, locoex::FusedActFunc::NONE);
  init(add_y);

  // let's run fusion
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::FuseBiasAddPass>();
    test_phase.run(g.graph());
  }

  auto a_conv2d = exo::test::find_first_node_bytype<locoex::TFLConv2D>(g.graph());
  ASSERT_TRUE(a_conv2d != nullptr);

  auto a_bias = dynamic_cast<locoex::TFLConst *>(a_conv2d->bias());
  ASSERT_TRUE(a_bias != nullptr);

  ASSERT_TRUE(a_bias->dim(0) == 2);
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(0),
                  bias->at<loco::DataType::FLOAT32>(0) + add_y->at<loco::DataType::FLOAT32>(0));
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(1),
                  bias->at<loco::DataType::FLOAT32>(1) + add_y->at<loco::DataType::FLOAT32>(1));
}

// A case when
// - TFLConv2D has bias (0, 0)
// - TFLAdd, of which x() or y() == TFLConv2D
// - Another param of TFLAdd is TFLConst, (1.5) <-- scalar
//
// After fusion, bias shold be (1.5, 1.5)
TEST(FuseBiasAddPassTest, Conv2D_Add_02_TFLAdd_y_is_scalar)
{
  exo::test::TestGraph g;
  auto filter = g.append<locoex::TFLConst>();
  auto bias = g.append<locoex::TFLConst>();
  auto conv2d = g.append<locoex::TFLConv2D>(g.pull, filter, bias);

  auto add_y = g.append<locoex::TFLConst>();
  auto add = g.append<locoex::TFLAdd>(conv2d, add_y);

  g.complete(add);

  init(g.pull);
  init(conv2d, filter, bias); // channel of conv2d is 2

  {
    // Size of this TFLConst is 1.
    // Note that this should be widened later to the shape of [channel of Conv2D], which is [2]
    add_y->dtype(loco::DataType::FLOAT32);
    add_y->shape({1});
    add_y->size<loco::DataType::FLOAT32>(1);
    add_y->at<loco::DataType::FLOAT32>(0) = 1.5;
  }
  init(add, locoex::FusedActFunc::NONE);

  // let's run fusion
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::FuseBiasAddPass>();
    test_phase.run(g.graph());
  }

  auto a_conv2d = exo::test::find_first_node_bytype<locoex::TFLConv2D>(g.graph());
  ASSERT_TRUE(a_conv2d != nullptr);

  auto a_bias = dynamic_cast<locoex::TFLConst *>(a_conv2d->bias());
  ASSERT_TRUE(a_bias != nullptr);

  ASSERT_TRUE(a_bias->dim(0) == 2);
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(0),
                  bias->at<loco::DataType::FLOAT32>(0) + 1.5);
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(1),
                  bias->at<loco::DataType::FLOAT32>(1) + 1.5);
}

// A case when
// - TFLConv2D has bias (0, 0)
// - TFLSub.x() == TFLConv2D
// - TFLSub.y() == TFLConst, (1.5, 3)
//
// After fusion, bias shold be (-1.5, -3)
TEST(FuseBiasAddPassTest, Conv2D_Sub_01_basic)
{
  exo::test::TestGraph g;
  auto filter = g.append<locoex::TFLConst>();
  auto bias = g.append<locoex::TFLConst>();
  auto conv2d = g.append<locoex::TFLConv2D>(g.pull, filter, bias);

  auto sub_y = g.append<locoex::TFLConst>();
  auto sub = g.append<locoex::TFLSub>(conv2d, sub_y);

  g.complete(sub);

  init(g.pull);
  init(conv2d, filter, bias);
  init(sub, locoex::FusedActFunc::NONE);
  init(sub_y);

  // let's run fusion
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::FuseBiasAddPass>();
    test_phase.run(g.graph());
  }

  auto a_conv2d = exo::test::find_first_node_bytype<locoex::TFLConv2D>(g.graph());
  ASSERT_TRUE(a_conv2d != nullptr);

  auto a_bias = dynamic_cast<locoex::TFLConst *>(a_conv2d->bias());
  ASSERT_TRUE(a_bias != nullptr);

  ASSERT_TRUE(a_bias->dim(0) == 2);
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(0),
                  bias->at<loco::DataType::FLOAT32>(0) - sub_y->at<loco::DataType::FLOAT32>(0));
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(1),
                  bias->at<loco::DataType::FLOAT32>(1) - sub_y->at<loco::DataType::FLOAT32>(1));
}

// A case when TFLConv2D is input of TFLSub but fusion cannot be performed.
// - TFLSub.x() == TFLConst
// - TFLSub.y() == TFLConv2D
//
// Here, TFLSub cannot be fused into TFLConst. To be fused, TFLSub.x() should be TFLConv2D and
// TFLSub.y() should be TFLConst. So fusion will NOT happen.
TEST(FuseBiasAddPassTest, Conv2D_Sub_02_fusing_will_not_performed)
{
  exo::test::TestGraph g;
  auto filter = g.append<locoex::TFLConst>();
  auto bias = g.append<locoex::TFLConst>();
  auto conv2d = g.append<locoex::TFLConv2D>(g.pull, filter, bias);

  auto sub_y = g.append<locoex::TFLConst>();
  auto sub = g.append<locoex::TFLSub>(sub_y, conv2d); // This WON'T be fused

  g.complete(sub);

  init(g.pull);
  init(conv2d, filter, bias);
  init(sub, locoex::FusedActFunc::NONE);
  init(sub_y);

  // let's run fusion
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::FuseBiasAddPass>();
    test_phase.run(g.graph());
  }

  auto a_conv2d = exo::test::find_first_node_bytype<locoex::TFLConv2D>(g.graph());
  ASSERT_TRUE(a_conv2d != nullptr);

  auto a_bias = dynamic_cast<locoex::TFLConst *>(a_conv2d->bias());
  ASSERT_TRUE(a_bias != nullptr);

  ASSERT_TRUE(a_bias->dim(0) == 2);
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(0), 0);
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(1), 0);

  auto a_sub = exo::test::find_first_node_bytype<locoex::TFLSub>(g.graph());
  ASSERT_TRUE(a_sub != nullptr);
  ASSERT_TRUE(a_sub->y() == a_conv2d); // Checking 'not-fused' state
}

// A case when
// - TFLConv2D has an activation function with Relu
// - TFLAdd, has no activation function
//
// No fusion should happen
TEST(FuseBiasAddPassTest, Regression_Conv2D_Add_fused_action_00)
{
  exo::test::TestGraph g;
  auto filter = g.append<locoex::TFLConst>();
  auto bias = g.append<locoex::TFLConst>();
  auto conv2d = g.append<locoex::TFLConv2D>(g.pull, filter, bias);

  auto add_y = g.append<locoex::TFLConst>();
  auto add = g.append<locoex::TFLAdd>(conv2d, add_y);

  g.complete(add);

  init(g.pull);
  init(conv2d, filter, bias);
  init(add, locoex::FusedActFunc::NONE);
  init(add_y);

  // Updating Fused Activation for this test
  conv2d->fusedActivationFunction(locoex::FusedActFunc::RELU);

  // let's run fusion
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::FuseBiasAddPass>();
    test_phase.run(g.graph());
  }

  auto a_conv2d = exo::test::find_first_node_bytype<locoex::TFLConv2D>(g.graph());
  ASSERT_TRUE(a_conv2d != nullptr);
  ASSERT_TRUE(a_conv2d->fusedActivationFunction() == locoex::FusedActFunc::RELU);

  auto an_add = exo::test::find_first_node_bytype<locoex::TFLAdd>(g.graph());
  ASSERT_TRUE(an_add != nullptr);
  ASSERT_TRUE(an_add->fusedActivationFunction() == locoex::FusedActFunc::NONE);

  ASSERT_TRUE(an_add->x() == a_conv2d or an_add->y() == a_conv2d);
}

// A case when
// - TFLConv2D has NONE activation function
// - TFLAdd has Relu activation function
//
// TFLConv2D should have Relu activation function, TFLAdd is fused into bias input
TEST(FuseBiasAddPassTest, Regression_Conv2D_Add_fused_action_01)
{
  exo::test::TestGraph g;
  auto filter = g.append<locoex::TFLConst>();
  auto bias = g.append<locoex::TFLConst>();
  auto conv2d = g.append<locoex::TFLConv2D>(g.pull, filter, bias);

  auto add_y = g.append<locoex::TFLConst>();
  auto add = g.append<locoex::TFLAdd>(conv2d, add_y);

  g.complete(add);

  init(g.pull);
  init(conv2d, filter, bias);
  init(add, locoex::FusedActFunc::RELU);
  init(add_y);

  // let's run fusion
  {
    exo::test::TypeShapeReadyPhase test_phase;

    test_phase.add_pass<exo::FuseBiasAddPass>();
    test_phase.run(g.graph());
  }

  auto a_conv2d = exo::test::find_first_node_bytype<locoex::TFLConv2D>(g.graph());
  ASSERT_TRUE(a_conv2d != nullptr);

  auto a_bias = dynamic_cast<locoex::TFLConst *>(a_conv2d->bias());
  ASSERT_TRUE(a_bias != nullptr);

  ASSERT_TRUE(a_bias->dim(0) == 2);
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(0),
                  bias->at<loco::DataType::FLOAT32>(0) + add_y->at<loco::DataType::FLOAT32>(0));
  ASSERT_FLOAT_EQ(a_bias->at<loco::DataType::FLOAT32>(1),
                  bias->at<loco::DataType::FLOAT32>(1) + add_y->at<loco::DataType::FLOAT32>(1));

  ASSERT_TRUE(a_conv2d->fusedActivationFunction() == locoex::FusedActFunc::RELU);
}
