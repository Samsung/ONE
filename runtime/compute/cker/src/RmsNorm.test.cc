/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/operation/RmsNorm.h>

#include <gtest/gtest.h>
#include <vector>

TEST(CKer_Operation, RmsNorm)
{
  // Simple
  {
    std::vector<float> input = {0, 1, 2, 3};
    nnfw::cker::Shape input_shape{1, 2, 2, 1};

    std::vector<float> expected_output = {0, 1, 1, 1};
    std::vector<float> output(expected_output.size());
    nnfw::cker::Shape output_shape{1, 2, 2, 1};

    std::vector<float> gamma = {1};
    nnfw::cker::Shape gamma_shape{1};

    nnfw::cker::RmsNormParams param;
    param.epsilon = 0.00001f;

    nnfw::cker::RmsNorm(param, input_shape, input.data(), gamma_shape, gamma.data(), output_shape,
                        output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_NEAR(output[i], expected_output[i], 1e-5f);
  }

  // rank 4
  {
    std::vector<float> input = {0, 1, 2, 3, 4, 5, 6, 7};
    nnfw::cker::Shape input_shape{1, 2, 2, 2};

    std::vector<float> expected_output = {0,        1.412802, 0.784404, 1.176606,
                                          0.883431, 1.104288, 0.920347, 1.073738};
    std::vector<float> output(expected_output.size());
    nnfw::cker::Shape output_shape{1, 2, 2, 2};

    std::vector<float> gamma = {1, 1};
    nnfw::cker::Shape gamma_shape{2};

    nnfw::cker::RmsNormParams param;
    param.epsilon = 0.001f;

    nnfw::cker::RmsNorm(param, input_shape, input.data(), gamma_shape, gamma.data(), output_shape,
                        output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_NEAR(output[i], expected_output[i], 1e-5f);
  }

  // rank 3
  {
    std::vector<float> input = {0, 1, 2, 3, 4, 5, 6, 7};
    nnfw::cker::Shape input_shape{2, 2, 2};

    std::vector<float> expected_output = {0,        1.412802, 0.784404, 1.176606,
                                          0.883431, 1.104288, 0.920347, 1.073738};
    std::vector<float> output(expected_output.size());
    nnfw::cker::Shape output_shape{2, 2, 2};

    std::vector<float> gamma = {1, 1};
    nnfw::cker::Shape gamma_shape{2};

    nnfw::cker::RmsNormParams param;
    param.epsilon = 0.001f;

    nnfw::cker::RmsNorm(param, input_shape, input.data(), gamma_shape, gamma.data(), output_shape,
                        output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_NEAR(output[i], expected_output[i], 1e-5f);
  }
}

TEST(CKer_Operation, neg_RmsNormWrongInputDims)
{
  {
    std::vector<float> input = {0, 1, 2, 3};
    nnfw::cker::Shape input_shape{2, 2};

    std::vector<float> expected_output = {0, 1, 1, 1};
    std::vector<float> output(expected_output.size());
    nnfw::cker::Shape output_shape{2, 2};

    std::vector<float> gamma = {1};
    nnfw::cker::Shape gamma_shape{1};

    nnfw::cker::RmsNormParams param;
    param.epsilon = 0.00001f;

    EXPECT_ANY_THROW(nnfw::cker::RmsNorm(param, input_shape, input.data(), gamma_shape,
                                         gamma.data(), output_shape, output.data()));
  }
}
