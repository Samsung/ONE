/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/operation/GELU.h>

#include <gtest/gtest.h>
#include <vector>

TEST(CKer_Operation, GELU)
{
  // Approximate = true
  {
    std::vector<float> input = {-3, -2, -1, 0, 1, 2, 3};
    nnfw::cker::Shape input_shape{7};

    std::vector<float> expected_output = {-0.0036, -0.0454, -0.1588, 0, 0.8412, 1.9546, 2.9964};
    std::vector<float> output(expected_output.size());
    nnfw::cker::Shape output_shape{7};

    nnfw::cker::GELUParams param;
    param.approximate = true;

    nnfw::cker::GELU(param, input_shape, input.data(), output_shape, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_NEAR(output[i], expected_output[i], 1e-4f);
  }

  // Approximate = false
  {
    std::vector<float> input = {-3, -2, -1, 0, 1, 2, 3};
    nnfw::cker::Shape input_shape{7};

    std::vector<float> expected_output = {-0.0040, -0.0455, -0.1587, 0, 0.8413, 1.9545, 2.9960};
    std::vector<float> output(expected_output.size());
    nnfw::cker::Shape output_shape{7};

    nnfw::cker::GELUParams param;
    param.approximate = false;

    nnfw::cker::GELU(param, input_shape, input.data(), output_shape, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_NEAR(output[i], expected_output[i], 1e-4f);
  }
}
