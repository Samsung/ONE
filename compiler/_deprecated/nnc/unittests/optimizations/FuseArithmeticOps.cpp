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

#include "passes/optimizations/FuseArithmeticOps.h"
#include "Util.h"
#include "mir/Graph.h"
#include "mir/ops/AddOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/MulOp.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace nnc;
using namespace mir;

namespace
{

TEST(OptPass, fuseConvBiasScaleScaleBias)
{
  mir::Graph g;

  // Create graph: 'input->conv->bias->scale->scale->bias'
  mir::TensorType input_type(mir::DataType::FLOAT32, Shape{1, 299, 299, 3});
  auto input = g.create<ops::InputOp>(input_type);
  auto conv_const = g.create<ops::ConstantOp>(TensorVariant(DataType::FLOAT32, {10, 3, 3, 3}));
  auto conv = g.create<ops::Conv2DOp>(input->getOutput(0), conv_const->getOutput(0),
                                      mir::Conv2DOpAttributes());
  auto bias1_const = g.create<ops::ConstantOp>(TensorVariant(DataType::FLOAT32, {10}));
  auto bias1 = g.create<ops::AddOp>(conv->getOutput(0), bias1_const->getOutput(0));
  auto scale1_const = g.create<ops::ConstantOp>(TensorVariant(DataType::FLOAT32, {10}));
  auto scale1 = g.create<ops::MulOp>(bias1->getOutput(0), scale1_const->getOutput(0));
  auto scale2_const = g.create<ops::ConstantOp>(TensorVariant(DataType::FLOAT32, {10}));
  auto scale2 = g.create<ops::MulOp>(scale1->getOutput(0), scale2_const->getOutput(0));
  auto scale3_const = g.create<ops::ConstantOp>(TensorVariant(DataType::FLOAT32, {10}));
  auto scale3 = g.create<ops::MulOp>(scale2->getOutput(0), scale3_const->getOutput(0));
  auto bias2_const = g.create<ops::ConstantOp>(TensorVariant(DataType::FLOAT32, {10}));
  g.create<ops::AddOp>(scale3->getOutput(0), bias2_const->getOutput(0));

  // Check that layout is desired
  std::stringstream ss;
  DumpVisitor d(ss);
  FuseArithmeticOps pass;
  pass.run(&g);
  g.accept(&d);
  // Assert only 'conv->bias' remains
  ASSERT_TRUE("i_0.const_25.const_23.conv_26.b_24." == ss.str() ||
              "i_0.const_23.const_25.conv_26.b_24." == ss.str() ||
              "const_25.i_0.const_23.conv_26.b_24." == ss.str() ||
              "const_23.i_0.const_25.conv_26.b_24." == ss.str() ||
              "const_25.const_23.i_0.conv_26.b_24." == ss.str() ||
              "const_23.const_25.i_0.conv_26.b_24." == ss.str());
}

} // unnamed namespace
