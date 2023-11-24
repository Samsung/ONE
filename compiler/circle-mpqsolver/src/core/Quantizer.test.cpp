/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <gtest/gtest.h>

#include "Quantizer.h"
#include "TestHelper.h"

#include <luci/IR/CircleNodes.h>

using namespace mpqsolver::test::models;

TEST(CircleMPQSolverQuantizerTest, verifyResultsTest)
{
  auto m = luci::make_module();
  AddGraph g;
  g.init();
  auto add = g._add;
  float range = g._a_max - g._a_min;
  g.transfer_to(m.get());

  mpqsolver::core::Quantizer::Context context;
  mpqsolver::core::Quantizer quantizer(context);
  mpqsolver::core::LayerParams params;
  auto res = quantizer.quantize(m.get(), context.output_model_dtype, params);
  EXPECT_TRUE(res);
  auto quant_param = add->quantparam();
  EXPECT_TRUE(quant_param != nullptr);
  EXPECT_TRUE(quant_param->scale.size() == 1);
  EXPECT_FLOAT_EQ(quant_param->scale[0], range / 255.f);
  EXPECT_TRUE(quant_param->zerop.size() == 1);
  EXPECT_TRUE(quant_param->zerop[0] == 128);
}

TEST(CircleMPQSolverQuantizerTest, verifyResultsTest_NEG)
{
  mpqsolver::core::Quantizer::Context context;
  mpqsolver::core::Quantizer quantizer(context);
  mpqsolver::core::LayerParams params;
  auto res = quantizer.quantize(nullptr, context.output_model_dtype, params);
  EXPECT_TRUE(!res);
}
