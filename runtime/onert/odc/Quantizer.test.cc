/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Quantizer.h"

#include <gtest/gtest.h>

using namespace onert::odc;

// Test model input path is not set
TEST(odc_Quantizer, neg_model_input_path)
{
  Quantizer quantizer;
  ASSERT_THROW(quantizer.quantize(nullptr, "out", false), std::runtime_error);
}

// Test model output path is not set
TEST(odc_Quantizer, neg_model_output_path)
{
  Quantizer quantizer;
  ASSERT_THROW(quantizer.quantize("in", nullptr, false), std::runtime_error);
}

// Test invalid model input path
TEST(odc_Quantizer, neg_invalid_model_input_path)
{
  Quantizer quantizer;
  ASSERT_EQ(quantizer.quantize("invalid_model_input_path.circle", "out", false), -1);
}
