/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/QuantizeWeightsPass.h"

#include <gtest/gtest.h>

TEST(QuantizeWeightsPassTest, name)
{
  luci::QuantizeWeightsPass pass(loco::DataType::FLOAT32, loco::DataType::U8,
                                 luci::QuantizationGranularity::LayerWise);
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(QuantizeWeightsPassTest, name_ctx)
{
  auto ctx = std::make_unique<luci::QuantizeWeightsPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = loco::DataType::U8;
    ctx->granularity = luci::QuantizationGranularity::LayerWise;
  }

  luci::QuantizeWeightsPass pass(std::move(ctx));
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}
